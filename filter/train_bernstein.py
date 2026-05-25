"""
Train and export HGC Bernstein-filtered item embeddings.

Pipeline:
1. Load or extract 768-dim textual features for HGC items
2. Build an item-item co-occurrence graph from train interactions
3. Apply per-item Top-K truncation before building the normalized Laplacian
4. Precompute Bernstein basis responses with sparse SpMM
5. Train non-negative Bernstein coefficients with InfoNCE + preserve loss
6. Export filtered embeddings as a full item matrix plus metadata
"""

import argparse
import json
import logging
import math
import os
import pickle
import random
import warnings
from collections import defaultdict
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy import sparse
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from transformers import AutoConfig, AutoModel, AutoTokenizer

warnings.filterwarnings("ignore")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class BertClsExtractor:
    def __init__(self, plm_model: str = "bert-base-uncased", device: str = "cpu", max_length: int = 512):
        self.device = device
        self.max_length = max_length

        logging.info(f"Loading PLM: {plm_model}")
        self.tokenizer = AutoTokenizer.from_pretrained(plm_model)
        config = AutoConfig.from_pretrained(plm_model, output_hidden_states=True)
        self.bert_model = AutoModel.from_pretrained(plm_model, config=config)
        self.bert_model.eval().to(self.device)
        for param in self.bert_model.parameters():
            param.requires_grad = False
        self.embedding_dim = config.hidden_size

    def extract_from_text_file(self, text_file: str, batch_size: int = 64) -> Dict[str, np.ndarray]:
        df = pd.read_csv(text_file, sep="\t", header=0, dtype={0: str})
        df.columns = ["ITEM", "TEXT"]
        df["ITEM"] = df["ITEM"].astype(str)
        df["TEXT"] = df["TEXT"].astype(str).fillna("")

        item_ids = df["ITEM"].tolist()
        texts = df["TEXT"].tolist()
        item2vector: Dict[str, np.ndarray] = {}
        num_batches = (len(texts) + batch_size - 1) // batch_size

        for batch_idx in tqdm(range(num_batches), desc=f"[CLS] {os.path.basename(text_file)}"):
            start = batch_idx * batch_size
            end = min(start + batch_size, len(texts))
            encoded = self.tokenizer(
                texts[start:end],
                max_length=self.max_length,
                padding="max_length",
                truncation=True,
                return_tensors="pt",
            )
            input_ids = encoded["input_ids"].to(self.device)
            attention_mask = encoded["attention_mask"].to(self.device)
            token_type_ids = encoded.get("token_type_ids", torch.zeros_like(input_ids)).to(self.device)

            with torch.no_grad():
                outputs = self.bert_model(
                    input_ids=input_ids,
                    token_type_ids=token_type_ids,
                    attention_mask=attention_mask,
                )
                cls_vecs = outputs.last_hidden_state[:, 0, :].cpu().numpy()

            for item_id, vec in zip(item_ids[start:end], cls_vecs):
                item2vector[item_id] = vec

        logging.info(f"Extracted {len(item2vector)} textual embeddings from {text_file}")
        return item2vector


def load_item_features(
    path: str,
    batch_size: int,
    plm_model: str,
    device: str,
    max_length: int,
) -> Dict[str, np.ndarray]:
    suffix = os.path.splitext(path)[1].lower()
    if suffix == ".tsv":
        extractor = BertClsExtractor(plm_model=plm_model, device=device, max_length=max_length)
        return extractor.extract_from_text_file(path, batch_size=batch_size)

    if suffix == ".pkl":
        with open(path, "rb") as file_obj:
            feats = pickle.load(file_obj)
    elif suffix in (".pt", ".pth"):
        feats = torch.load(path, map_location="cpu")
    elif suffix == ".npy":
        feats = np.load(path)
    else:
        raise ValueError(f"Unsupported feature input suffix: {suffix}")

    if isinstance(feats, torch.Tensor):
        feats = feats.cpu().numpy()
    feats = np.asarray(feats, dtype=np.float32)
    return {str(idx): row for idx, row in enumerate(feats)}


def build_cooccurrence_graph(train_file: str, window_size: int = 2) -> Tuple[Dict[Tuple[str, str], int], set]:
    df = pd.read_csv(train_file, sep="\t", header=0)
    required_cols = ["USER", "ITEM", "TIMESTAMP"]
    if not set(required_cols).issubset(df.columns):
        if df.shape[1] < 4:
            raise ValueError(
                f"train_file={train_file} must contain at least USER/ITEM/TIMESTAMP information; "
                f"got columns={list(df.columns)}"
            )
        # Backward compatibility for older 4-column exports without stable headers.
        df = df.iloc[:, :4].copy()
        df.columns = ["USER", "ITEM", "RATING", "TIMESTAMP"]
    else:
        # Some processed datasets (for example Yelp) include extra review metadata columns.
        keep_cols = ["USER", "ITEM", "TIMESTAMP"]
        if "RATING" in df.columns:
            keep_cols.insert(2, "RATING")
        df = df[keep_cols].copy()
        if "RATING" not in df.columns:
            df["RATING"] = 1.0

    df["USER"] = df["USER"].astype(str)
    df["ITEM"] = df["ITEM"].astype(str)

    edge_counts = defaultdict(int)
    all_items = set()
    user_groups = df.sort_values(["USER", "TIMESTAMP"]).groupby("USER")

    for _, group in tqdm(user_groups, desc="Building graph"):
        items = group["ITEM"].tolist()
        all_items.update(items)
        if len(items) < 2:
            continue

        for center in range(len(items)):
            left = max(0, center - window_size)
            right = min(len(items), center + window_size + 1)
            for neighbor_idx in range(left, right):
                if neighbor_idx == center:
                    continue
                u, v = items[center], items[neighbor_idx]
                if u == v:
                    continue
                edge_counts[(min(u, v), max(u, v))] += 1

    logging.info(f"Graph before truncation: {len(all_items)} items, {len(edge_counts)} undirected edges")
    return dict(edge_counts), all_items


def truncate_edges(
    edge_counts: Dict[Tuple[str, str], int],
    item_list: List[str],
    topk_neighbors: int,
    mode: str = "topk",
    count_threshold: int = 2,
) -> Dict[Tuple[str, str], int]:
    if mode not in ("topk", "threshold", "topk_and_threshold"):
        raise ValueError(f"Unknown truncate_mode: {mode}")

    if mode == "topk" and topk_neighbors <= 0:
        return edge_counts

    result = dict(edge_counts)

    # Step 1: top-K per node (used by topk and topk_and_threshold modes)
    if mode in ("topk", "topk_and_threshold") and topk_neighbors > 0:
        adjacency = defaultdict(list)
        for (u, v), count in edge_counts.items():
            adjacency[u].append((v, count))
            adjacency[v].append((u, count))

        keep_directed = set()
        for item in item_list:
            neighbors = adjacency.get(item, [])
            if not neighbors:
                continue
            neighbors = sorted(neighbors, key=lambda x: (-x[1], x[0]))[:topk_neighbors]
            for neighbor, _ in neighbors:
                keep_directed.add((item, neighbor))

        topk_result = {}
        for (u, v), count in edge_counts.items():
            if (u, v) in keep_directed or (v, u) in keep_directed:
                topk_result[(u, v)] = count
        result = topk_result

    # Step 2: count threshold (used by threshold and topk_and_threshold modes)
    if mode in ("threshold", "topk_and_threshold"):
        result = {
            (u, v): count for (u, v), count in result.items()
            if count >= count_threshold
        }

    mode_desc = {
        "topk": f"Top-{topk_neighbors}",
        "threshold": f"count>={count_threshold}",
        "topk_and_threshold": f"Top-{topk_neighbors} AND count>={count_threshold}",
    }
    logging.info(
        f"Graph after truncation ({mode_desc[mode]}): {len(result)} undirected edges "
        f"(from {len(edge_counts)})"
    )
    return result


def build_adjacency_matrix(edge_counts: Dict[Tuple[str, str], int], item_list: List[str]):
    item_to_idx = {item: idx for idx, item in enumerate(item_list)}
    num_items = len(item_list)
    rows, cols, data = [], [], []

    for (u, v), count in edge_counts.items():
        if u in item_to_idx and v in item_to_idx:
            i, j = item_to_idx[u], item_to_idx[v]
            rows.extend([i, j])
            cols.extend([j, i])
            data.extend([count, count])

    adjacency = sparse.csr_matrix((data, (rows, cols)), shape=(num_items, num_items), dtype=np.float32)
    degrees = np.asarray(adjacency.sum(axis=1)).flatten()
    active_mask = degrees > 0
    active_indices = np.where(active_mask)[0]
    active_items = [item_list[i] for i in active_indices]
    adjacency_active = adjacency[active_mask][:, active_mask].tocsr()

    logging.info(f"Active items (degree>0): {len(active_items)} / {num_items}")
    return adjacency_active, active_items, active_indices


def build_normalized_laplacian(adjacency: sparse.csr_matrix, eps: float = 1e-12) -> sparse.csr_matrix:
    degrees = np.asarray(adjacency.sum(axis=1)).flatten()
    d_inv_sqrt = np.zeros_like(degrees, dtype=np.float32)
    mask = degrees > eps
    d_inv_sqrt[mask] = 1.0 / np.sqrt(degrees[mask] + eps)
    d_mat = sparse.diags(d_inv_sqrt)
    num_items = adjacency.shape[0]
    return sparse.eye(num_items, format="csr", dtype=np.float32) - d_mat @ adjacency @ d_mat


def scipy_csr_to_torch_sparse(matrix: sparse.csr_matrix, device: torch.device) -> torch.Tensor:
    coo = matrix.tocoo()
    indices = torch.from_numpy(np.vstack([coo.row, coo.col]).astype(np.int64))
    values = torch.from_numpy(coo.data.astype(np.float32))
    shape = torch.Size(coo.shape)
    return torch.sparse_coo_tensor(indices, values, shape, device=device).coalesce()


def build_embedding_matrices(emb_dict: Dict[str, np.ndarray], active_items: List[str]):
    all_item_ids_int = sorted(int(k) for k in emb_dict.keys())
    num_items = max(all_item_ids_int) + 1
    embedding_dim = next(iter(emb_dict.values())).shape[0]

    x_all = np.zeros((num_items, embedding_dim), dtype=np.float32)
    valid_mask = np.zeros(num_items, dtype=bool)
    for item_id_str, vec in emb_dict.items():
        idx = int(item_id_str)
        x_all[idx] = vec
        valid_mask[idx] = True

    global_mean = x_all[valid_mask].mean(axis=0, keepdims=True)
    x_all[valid_mask] -= global_mean

    active_indices = np.array([int(item) for item in active_items], dtype=np.int64)
    x_active = x_all[active_indices].copy()

    return {
        "num_items": num_items,
        "embedding_dim": embedding_dim,
        "valid_mask": valid_mask,
        "n_valid": int(valid_mask.sum()),
        "global_mean": global_mean,
        "active_indices": active_indices,
        "x_all_centered": x_all,
        "x_active_centered": x_active,
    }


def build_neighbor_lists(adjacency: sparse.csr_matrix) -> List[np.ndarray]:
    neighbors = []
    for row_idx in range(adjacency.shape[0]):
        start = adjacency.indptr[row_idx]
        end = adjacency.indptr[row_idx + 1]
        row_neighbors = adjacency.indices[start:end]
        row_neighbors = row_neighbors[row_neighbors != row_idx]
        neighbors.append(row_neighbors.astype(np.int64, copy=False))
    return neighbors


def precompute_bernstein_basis(
    laplacian: torch.Tensor,
    inputs: torch.Tensor,
    order_k: int,
) -> torch.Tensor:
    num_nodes = inputs.size(0)
    identity_indices = torch.arange(num_nodes, device=inputs.device)
    identity = torch.sparse_coo_tensor(
        torch.stack([identity_indices, identity_indices]),
        torch.ones(num_nodes, device=inputs.device),
        (num_nodes, num_nodes),
        device=inputs.device,
    ).coalesce()

    p_op = (identity - 0.5 * laplacian).coalesce()
    q_op = (0.5 * laplacian).coalesce()

    q_applied = [inputs]
    for _ in range(order_k):
        q_applied.append(torch.sparse.mm(q_op, q_applied[-1]))

    basis = []
    for k in range(order_k + 1):
        term = q_applied[order_k - k]
        for _ in range(k):
            term = torch.sparse.mm(p_op, term)
        coeff = float(math.comb(order_k, k))
        basis.append(term * coeff)

    return torch.stack(basis, dim=0)


class PositivePairDataset(Dataset):
    def __init__(self, neighbor_lists: List[np.ndarray], seed: int):
        self.anchor_indices = [idx for idx, neighbors in enumerate(neighbor_lists) if len(neighbors) > 0]
        self.neighbor_lists = neighbor_lists
        self.rng = np.random.default_rng(seed)

    def __len__(self) -> int:
        return len(self.anchor_indices)

    def __getitem__(self, index: int) -> Tuple[int, int]:
        anchor = self.anchor_indices[index]
        neighbors = self.neighbor_lists[anchor]
        positive = int(neighbors[self.rng.integers(0, len(neighbors))])
        return anchor, positive


class BernsteinFilter(nn.Module):
    def __init__(self, basis_outputs: torch.Tensor, init_value: float = 0.0, coeff_eps: float = 1e-8):
        super().__init__()
        self.register_buffer("basis_outputs", basis_outputs)
        self.raw_coeffs = nn.Parameter(torch.full((basis_outputs.size(0),), float(init_value)))
        self.coeff_eps = coeff_eps

    def positive_coeffs(self) -> torch.Tensor:
        return F.softplus(self.raw_coeffs) + self.coeff_eps

    def filtered_embeddings(self) -> torch.Tensor:
        coeffs = self.positive_coeffs().view(-1, 1, 1)
        return torch.sum(coeffs * self.basis_outputs, dim=0)


def evaluate_recommendation(
    filtered_full: torch.Tensor,
    train_file: str,
    valid_file: str,
    ks: List[int] = [10, 20],
) -> Dict[str, float]:
    """LightGCN-style frozen-embedding evaluation on validation set."""
    train_df = pd.read_csv(train_file, sep="\t", header=0)
    valid_df = pd.read_csv(valid_file, sep="\t", header=0)

    train_df["USER"] = train_df["USER"].astype(str)
    train_df["ITEM"] = train_df["ITEM"].astype(str)
    valid_df["USER"] = valid_df["USER"].astype(str)
    valid_df["ITEM"] = valid_df["ITEM"].astype(str)

    all_users = sorted(set(train_df["USER"].unique()) | set(valid_df["USER"].unique()))
    all_items = sorted(set(train_df["ITEM"].unique()) | set(valid_df["ITEM"].unique()))
    user2idx = {u: i for i, u in enumerate(all_users)}
    item2idx = {it: i for i, it in enumerate(all_items)}
    num_users = len(all_users)
    num_items = len(all_items)

    user_train_items = defaultdict(set)
    user_train_seqs = defaultdict(list)
    for _, row in train_df.iterrows():
        u = user2idx[row["USER"]]
        it = item2idx[row["ITEM"]]
        user_train_items[u].add(it)
        user_train_seqs[u].append(it)

    emb_dim = filtered_full.size(1)
    item_embs = torch.zeros(num_items, emb_dim, device=filtered_full.device)
    for item_str, idx in item2idx.items():
        src_idx = int(item_str)
        if src_idx < filtered_full.size(0):
            item_embs[idx] = filtered_full[src_idx]

    user_embs = torch.zeros(num_users, emb_dim, device=filtered_full.device)
    for u in range(num_users):
        seq = user_train_seqs.get(u, [])
        if seq:
            user_embs[u] = item_embs[torch.tensor(seq, device=filtered_full.device)].mean(dim=0)

    user_embs = F.normalize(user_embs, p=2, dim=1)
    item_embs = F.normalize(item_embs, p=2, dim=1)

    targets = []
    for _, row in valid_df.iterrows():
        targets.append((user2idx[row["USER"]], item2idx[row["ITEM"]]))
    target_users = torch.tensor([t[0] for t in targets], device=filtered_full.device)
    target_items = torch.tensor([t[1] for t in targets], device=filtered_full.device)

    max_k = max(ks)
    hr = {k: 0.0 for k in ks}
    ndcg = {k: 0.0 for k in ks}
    total = len(targets)

    batch_size_eval = 512
    for start in range(0, total, batch_size_eval):
        end = min(start + batch_size_eval, total)
        batch_u = target_users[start:end]
        batch_target = target_items[start:end]

        scores = user_embs[batch_u] @ item_embs.T  # (B, N)

        for i in range(batch_u.size(0)):
            u_idx = batch_u[i].item()
            train_items = user_train_items.get(u_idx, set())
            s = scores[i].clone()
            if train_items:
                s[list(train_items)] = -float("inf")

            _, topk_idx = s.topk(max_k)
            topk_list = topk_idx.tolist()
            target = batch_target[i].item()

            for k in ks:
                if target in topk_list[:k]:
                    hr[k] += 1.0
                    rank = topk_list[:k].index(target) + 1
                    ndcg[k] += 1.0 / math.log2(rank + 1)

    result = {}
    for k in ks:
        result[f"HR@{k}"] = hr[k] / max(total, 1)
        result[f"NDCG@{k}"] = ndcg[k] / max(total, 1)
    return result


def train_bernstein_filter(
    basis_outputs: torch.Tensor,
    original_feats: torch.Tensor,
    neighbor_lists: List[np.ndarray],
    alpha: float,
    temperature: float,
    epochs: int,
    batch_size: int,
    lr: float,
    weight_decay: float,
    seed: int,
    active_indices: np.ndarray = None,
    x_all_centered: np.ndarray = None,
    valid_mask: np.ndarray = None,
    train_file: str = "",
    valid_file: str = "",
    eval_every: int = 5,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    dataset = PositivePairDataset(neighbor_lists=neighbor_lists, seed=seed)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=False)
    model = BernsteinFilter(basis_outputs=basis_outputs).to(original_feats.device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    best_state = None
    best_val_state = None
    best_metrics = {
        "best_loss": float("inf"),
        "best_cl_loss": float("inf"),
        "best_preserve_loss": float("inf"),
        "best_val_ndcg10": 0.0,
        "best_val_epoch": -1,
        "best_epoch_by": "loss",
        "epochs_ran": 0,
    }

    do_eval = bool(valid_file) and active_indices is not None and x_all_centered is not None and valid_mask is not None

    logging.info(
        f"Start Bernstein training: num_active={original_feats.size(0)}, basis={basis_outputs.size(0)}, "
        f"alpha={alpha}, temperature={temperature}, epochs={epochs}, batch_size={batch_size}, "
        f"eval={do_eval}"
    )

    for epoch in range(1, epochs + 1):
        epoch_total = 0.0
        epoch_cl = 0.0
        epoch_preserve = 0.0
        epoch_steps = 0

        for anchor_idx, positive_idx in loader:
            anchor_idx = anchor_idx.to(original_feats.device)
            positive_idx = positive_idx.to(original_feats.device)

            filtered = model.filtered_embeddings()
            anchor_emb = filtered[anchor_idx]
            positive_emb = filtered[positive_idx]

            anchor_emb = F.normalize(anchor_emb, p=2, dim=1)
            positive_emb = F.normalize(positive_emb, p=2, dim=1)
            logits = anchor_emb @ positive_emb.t()
            logits = logits / temperature
            labels = torch.arange(logits.size(0), device=logits.device)
            cl_loss = F.cross_entropy(logits, labels)

            preserve_loss = F.mse_loss(filtered, original_feats)
            loss = cl_loss + alpha * preserve_loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_total += loss.item()
            epoch_cl += cl_loss.item()
            epoch_preserve += preserve_loss.item()
            epoch_steps += 1

        mean_total = epoch_total / max(epoch_steps, 1)
        mean_cl = epoch_cl / max(epoch_steps, 1)
        mean_preserve = epoch_preserve / max(epoch_steps, 1)
        best_metrics["epochs_ran"] = epoch

        logging.info(
            f"[Epoch {epoch:03d}] total={mean_total:.6f} "
            f"cl={mean_cl:.6f} preserve={mean_preserve:.6f} alpha*preserve={alpha * mean_preserve:.6f}"
        )

        if mean_total < best_metrics["best_loss"]:
            best_metrics["best_loss"] = mean_total
            best_metrics["best_cl_loss"] = mean_cl
            best_metrics["best_preserve_loss"] = mean_preserve
            best_state = {
                "raw_coeffs": model.raw_coeffs.detach().cpu().clone(),
            }

        if do_eval and epoch % eval_every == 0:
            with torch.no_grad():
                filtered_active = model.filtered_embeddings().detach().cpu().numpy()
            x_full = x_all_centered.copy()
            x_full[active_indices] = filtered_active
            x_tensor = torch.from_numpy(x_full).float()
            valid_indices = np.where(valid_mask)[0]
            x_tensor[torch.from_numpy(valid_indices)] = F.normalize(
                x_tensor[torch.from_numpy(valid_indices)], p=2, dim=1, eps=1e-12
            )
            val_metrics = evaluate_recommendation(x_tensor, train_file, valid_file)
            val_ndcg = val_metrics["NDCG@10"]
            if val_ndcg > best_metrics["best_val_ndcg10"]:
                best_metrics["best_val_ndcg10"] = val_ndcg
                best_metrics["best_val_epoch"] = epoch
                best_metrics["best_epoch_by"] = "val"
                best_val_state = {
                    "raw_coeffs": model.raw_coeffs.detach().cpu().clone(),
                }
                best_metrics["best_val_hr10"] = val_metrics["HR@10"]
            logging.info(
                f"[Epoch {epoch:03d} VAL] HR@10={val_metrics['HR@10']:.4f} "
                f"NDCG@10={val_ndcg:.4f} (best={best_metrics['best_val_ndcg10']:.4f} @{best_metrics['best_val_epoch']})"
            )

    if best_state is None and best_val_state is None:
        raise RuntimeError("Bernstein training did not produce a valid checkpoint")

    # Prefer validation-based checkpoint if available
    use_val = best_val_state is not None
    chosen_state = best_val_state if use_val else best_state
    chosen_epoch = best_metrics["best_val_epoch"] if use_val else best_metrics["epochs_ran"]

    with torch.no_grad():
        model.raw_coeffs.copy_(chosen_state["raw_coeffs"].to(model.raw_coeffs.device))
        filtered = model.filtered_embeddings().detach()
        coeffs = model.positive_coeffs().detach().cpu().numpy()

    best_metrics["coeff_min"] = float(coeffs.min())
    best_metrics["coeff_max"] = float(coeffs.max())
    best_metrics["chosen_epoch"] = chosen_epoch
    if use_val:
        logging.info(
            f"Bernstein checkpoint: epoch={chosen_epoch} selected by 'val', "
            f"val_ndcg@10={best_metrics['best_val_ndcg10']:.4f}"
        )
    else:
        logging.info(
            f"Bernstein checkpoint: epoch={chosen_epoch} selected by 'loss', "
            f"train_loss={best_metrics['best_loss']:.6f}"
        )
    return filtered, {**best_metrics, "coeffs": coeffs.tolist()}


def alpha_to_str(alpha: float) -> str:
    if alpha == 0:
        return "0"
    alpha_str = f"{alpha:g}"
    return alpha_str.replace(".", "p")


def save_filtered_embeddings(
    x_all_centered: np.ndarray,
    x_active_filtered: np.ndarray,
    valid_mask: np.ndarray,
    active_indices: np.ndarray,
    output_dir: str,
    bern_k: int,
    alpha: float,
    temperature: float,
    topk_neighbors: int,
    truncate_mode: str,
    count_threshold: int,
    window_size: int,
    n_active: int,
    n_edges_before_topk: int,
    n_edges_after_topk: int,
    global_mean: np.ndarray,
    metrics: Dict[str, float],
    text_max_length: int,
    eps: float = 1e-12,
) -> Tuple[str, str]:
    x_all_filtered = x_all_centered.copy()
    x_all_filtered[active_indices] = x_active_filtered

    x_tensor = torch.from_numpy(x_all_filtered).float()
    valid_indices = torch.from_numpy(np.where(valid_mask)[0])
    x_tensor[valid_indices] = F.normalize(x_tensor[valid_indices], p=2, dim=1, eps=eps)

    alpha_str = alpha_to_str(alpha)
    out_name = f"id_emb_HGC_bern_k{bern_k}_alpha{alpha_str}_hidden768.pkl"
    out_path = os.path.join(output_dir, out_name)
    with open(out_path, "wb") as file_obj:
        pickle.dump(x_tensor.numpy(), file_obj)

    meta = {
        "source": "HGC",
        "filter_family": "bernstein",
        "filter_impl": "trainable_bern",
        "bern_order": int(bern_k),
        "alpha": float(alpha),
        "temperature": float(temperature),
        "text_max_length": int(text_max_length),
        "truncate_mode": truncate_mode,
        "topk_neighbors": int(topk_neighbors),
        "count_threshold": int(count_threshold),
        "window_size": int(window_size),
        "n_items": int(x_all_centered.shape[0]),
        "n_valid": int(valid_mask.sum()),
        "n_active": int(n_active),
        "n_edges_before_topk": int(n_edges_before_topk),
        "n_edges_after_topk": int(n_edges_after_topk),
        "embedding_dim": int(x_all_centered.shape[1]),
        "global_mean_norm": float(np.linalg.norm(global_mean)),
        "best_loss": float(metrics["best_loss"]),
        "best_cl_loss": float(metrics["best_cl_loss"]),
        "best_preserve_loss": float(metrics["best_preserve_loss"]),
        "epochs_ran": int(metrics["epochs_ran"]),
        "chosen_epoch": int(metrics.get("chosen_epoch", metrics["epochs_ran"])),
        "best_epoch_by": metrics.get("best_epoch_by", "loss"),
        "best_val_ndcg10": float(metrics.get("best_val_ndcg10", 0.0)),
        "best_val_hr10": float(metrics.get("best_val_hr10", 0.0)),
        "best_val_epoch": int(metrics.get("best_val_epoch", -1)),
        "coeff_min": float(metrics["coeff_min"]),
        "coeff_max": float(metrics["coeff_max"]),
        "coeffs": metrics["coeffs"],
        "output_pkl": out_name,
    }

    meta_path = out_path.replace(".pkl", "_meta.json")
    with open(meta_path, "w") as file_obj:
        json.dump(meta, file_obj, indent=2)

    logging.info(f"Saved Bernstein embedding to {out_path}")
    return out_path, meta_path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="Amazon_Luxury_Beauty_2018")
    parser.add_argument("--source_type", type=str, default="hgc", choices=["hgc"])
    parser.add_argument("--text_file", type=str, required=True)
    parser.add_argument("--train_file", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--plm_model", type=str, default="bert-base-uncased")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--batch_size", type=int, default=512)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--lr", type=float, default=1e-2)
    parser.add_argument("--weight_decay", type=float, default=0.0)
    parser.add_argument("--max_length", type=int, default=512)
    parser.add_argument("--bern_k", type=int, required=True)
    parser.add_argument("--alpha", type=float, required=True)
    parser.add_argument("--temperature", type=float, default=0.1)
    parser.add_argument("--window_size", type=int, default=2)
    parser.add_argument("--topk_neighbors", type=int, default=20)
    parser.add_argument("--truncate_mode", type=str, default="topk",
                        choices=["topk", "threshold", "topk_and_threshold"])
    parser.add_argument("--count_threshold", type=int, default=3,
                        help="minimum co-occurrence count to retain an edge (for threshold/topk_and_threshold modes)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--valid_file", type=str, default="", help="path to valid.txt for LightGCN-style epoch selection")
    parser.add_argument("--eval_every", type=int, default=5, help="evaluate on validation set every N epochs")
    args = parser.parse_args()

    if args.temperature <= 0:
        raise ValueError("temperature must be > 0")
    if args.bern_k < 0:
        raise ValueError("bern_k must be >= 0")
    if args.alpha < 0:
        raise ValueError("alpha must be >= 0")
    if args.max_length <= 0:
        raise ValueError("max_length must be > 0")

    set_seed(args.seed)
    os.makedirs(args.output_dir, exist_ok=True)

    emb_dict = load_item_features(
        path=args.text_file,
        batch_size=args.batch_size,
        plm_model=args.plm_model,
        device=args.device,
        max_length=args.max_length,
    )

    edge_counts, graph_items = build_cooccurrence_graph(args.train_file, window_size=args.window_size)
    common_items = sorted(graph_items & set(emb_dict.keys()), key=int)
    logging.info(f"Intersection (graph ∩ source): {len(common_items)} items")

    edge_counts_truncated = truncate_edges(
        edge_counts=edge_counts,
        item_list=common_items,
        topk_neighbors=args.topk_neighbors,
        mode=args.truncate_mode,
        count_threshold=args.count_threshold,
    )

    adjacency_active, active_items, _ = build_adjacency_matrix(edge_counts_truncated, common_items)
    laplacian_active = build_normalized_laplacian(adjacency_active)
    neighbor_lists = build_neighbor_lists(adjacency_active)
    prepared = build_embedding_matrices(emb_dict, active_items)

    device = torch.device(args.device)
    x_active = torch.from_numpy(prepared["x_active_centered"]).to(device)
    laplacian_torch = scipy_csr_to_torch_sparse(laplacian_active, device=device)

    logging.info("Precomputing Bernstein basis responses with sparse SpMM")
    with torch.no_grad():
        basis_outputs = precompute_bernstein_basis(
            laplacian=laplacian_torch,
            inputs=x_active,
            order_k=args.bern_k,
        )

    filtered_active, metrics = train_bernstein_filter(
        basis_outputs=basis_outputs,
        original_feats=x_active,
        neighbor_lists=neighbor_lists,
        alpha=args.alpha,
        temperature=args.temperature,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        weight_decay=args.weight_decay,
        seed=args.seed,
        active_indices=prepared["active_indices"],
        x_all_centered=prepared["x_all_centered"],
        valid_mask=prepared["valid_mask"],
        train_file=args.train_file if args.valid_file else "",
        valid_file=args.valid_file,
        eval_every=args.eval_every,
    )

    save_filtered_embeddings(
        x_all_centered=prepared["x_all_centered"],
        x_active_filtered=filtered_active.detach().cpu().numpy(),
        valid_mask=prepared["valid_mask"],
        active_indices=prepared["active_indices"],
        output_dir=args.output_dir,
        bern_k=args.bern_k,
        alpha=args.alpha,
        temperature=args.temperature,
        topk_neighbors=args.topk_neighbors,
        truncate_mode=args.truncate_mode,
        count_threshold=args.count_threshold,
        window_size=args.window_size,
        n_active=len(active_items),
        n_edges_before_topk=len(edge_counts),
        n_edges_after_topk=len(edge_counts_truncated),
        global_mean=prepared["global_mean"],
        metrics=metrics,
        text_max_length=args.max_length,
    )

    logging.info("Done!")


if __name__ == "__main__":
    main()
