from typing import Dict, Tuple, Union
import json
import logging
import os
from collections import Counter

import freerec
import numpy as np
import torch
import torch.nn as nn
from freerec.utils import infoLogger
freerec.declare(version='1.0.1')


cfg = freerec.parser.Parser()
cfg.add_argument("--maxlen", type=int, default=50)
cfg.add_argument("--num-heads", type=int, default=1)
cfg.add_argument("--num-blocks", type=int, default=2)
cfg.add_argument("--embedding-dim", type=int, default=64)
cfg.add_argument("--dropout-rate", type=float, default=0.2)
cfg.add_argument("--loss", type=str, choices=('BPR', 'BCE', 'CE'), default='BCE')
cfg.add_argument("--tfile", type=str, default="", help="path to textual modality pkl file (relative to dataset dir); empty means use ID embeddings")
cfg.add_argument("--bucket-file", type=str, default="", help="path to item bucket JSON for per-bucket evaluation")
cfg.add_argument("--test-only", action="store_true", help="skip training, only run test evaluation")
cfg.add_argument("--ckpt-path", type=str, default="", help="path to best.pt checkpoint for test-only mode")
cfg.add_argument("--tfile-base", type=str, default="", help="path to base textual embedding file E^(0) (relative to dataset dir)")
cfg.add_argument("--tfile-low", type=str, default="", help="path to low-pass textual embedding file E_low (relative to dataset dir)")
cfg.add_argument(
    "--text-proj-mode",
    type=str,
    default="auto",
    choices=("auto", "linear", "identity", "frozen_pca"),
    help="how textual features are projected to embedding_dim",
)
cfg.add_argument(
    "--text-proj-pca-path",
    type=str,
    default="",
    help="path to frozen PCA projection params (.npz), relative to dataset dir unless absolute",
)
cfg.add_argument("--residual-alpha-init", type=float, default=1.0, help="initial scalar alpha for E_low + alpha * (E^(0) - E_low)")
cfg.add_argument("--learn-residual-alpha", type=eval, default=False, choices=(True, False), help="enable residual-alpha textual fusion mode")
cfg.add_argument("--gumbel-tau", type=float, default=0.0, help="Gumbel noise scale for BCE score regularization (0 = disabled)")
cfg.add_argument("--trainable-text", type=eval, default=False, choices=(True, False), help="make text_item_feats trainable instead of frozen buffer")
cfg.add_argument("--alpha-fuse", type=eval, default=False, choices=(True, False), help="AlphaFuse mode: SVD-whitened frozen signal + trainable null-space ID injection")
cfg.add_argument("--alpha-fuse-null-dim", type=int, default=64, help="number of null-space dimensions for AlphaFuse mode (must equal embedding-dim)")

cfg.set_defaults(
    description="SASRec",
    root="../../data",
    dataset='Amazon2014Beauty_550_LOU',
    epochs=200,
    batch_size=256,
    optimizer='adam',
    lr=1e-3,
    weight_decay=0.,
    seed=1,
)
cfg.compile()


def _load_item_features(path: str) -> torch.Tensor:
    suffix = os.path.splitext(path)[1].lower()

    if suffix == '.pkl':
        feats = freerec.utils.import_pickle(path)
    elif suffix in ('.pt', '.pth'):
        feats = torch.load(path, map_location='cpu')
    elif suffix == '.npy':
        feats = np.load(path)
    elif suffix == '.tsv':
        raise ValueError(
            f"tfile={path} is a TSV text file, not embedding vectors. "
            "Please provide a pre-encoded embedding file (.pkl/.pt/.npy)."
        )
    else:
        raise ValueError(f"Unsupported tfile suffix: {suffix}")

    if not isinstance(feats, torch.Tensor):
        feats = torch.as_tensor(feats)

    if feats.dim() != 2:
        raise ValueError(f"Expect textual feats with shape (num_items, dim), got {tuple(feats.shape)}")

    return feats.float()


def _resolve_feature_side_path(dataset_path: str, rel_or_abs_path: str) -> str:
    if not rel_or_abs_path:
        return ""
    if os.path.isabs(rel_or_abs_path):
        return rel_or_abs_path
    return os.path.join(dataset_path, rel_or_abs_path)


def _infer_frozen_pca_path(feature_path: str, embedding_dim: int) -> str:
    stem, _ = os.path.splitext(feature_path)
    return f"{stem}_frozen_pca{embedding_dim}.npz"


def _load_frozen_pca_params(path: str) -> Tuple[torch.Tensor, torch.Tensor]:
    payload = np.load(path)
    mean = torch.as_tensor(payload["mean"]).float()
    components = torch.as_tensor(payload["components"]).float()
    return mean, components


class FrozenPCALayer(nn.Module):

    def __init__(self, mean: torch.Tensor, components: torch.Tensor):
        super().__init__()
        self.register_buffer("mean", mean.view(1, -1))
        self.register_buffer("components", components)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return (inputs - self.mean) @ self.components


def _build_input_projection(
    feat_dim: int,
    embedding_dim: int,
    dataset_path: str,
    feature_path: str,
    proj_mode: str,
    pca_path: str = "",
    allow_infer_pca_path: bool = True,
) -> Tuple[nn.Module, str]:
    if proj_mode == "auto":
        if feat_dim == embedding_dim:
            return nn.Identity(), f"bypass in_proj because feat_dim == embedding_dim == {embedding_dim}"
        return nn.Linear(feat_dim, embedding_dim), f"project {feat_dim} -> {embedding_dim}"

    if proj_mode == "identity":
        if feat_dim != embedding_dim:
            raise ValueError(
                f"text_proj_mode=identity requires feat_dim == embedding_dim, got {feat_dim} vs {embedding_dim}"
            )
        return nn.Identity(), f"bypass in_proj because feat_dim == embedding_dim == {embedding_dim}"

    if proj_mode == "linear":
        return nn.Linear(feat_dim, embedding_dim), f"project {feat_dim} -> {embedding_dim}"

    if proj_mode == "frozen_pca":
        resolved_pca_path = _resolve_feature_side_path(dataset_path, pca_path)
        if not resolved_pca_path:
            if not allow_infer_pca_path:
                raise ValueError("text_proj_mode=frozen_pca requires --text-proj-pca-path in this mode")
            resolved_pca_path = _infer_frozen_pca_path(feature_path, embedding_dim)
        if not os.path.exists(resolved_pca_path):
            raise FileNotFoundError(f"Frozen PCA params not found: {resolved_pca_path}")

        mean, components = _load_frozen_pca_params(resolved_pca_path)
        if mean.numel() != feat_dim:
            raise ValueError(
                f"Frozen PCA mean dim mismatch: expect {feat_dim}, got {mean.numel()} from {resolved_pca_path}"
            )
        if components.dim() != 2 or components.size(0) != feat_dim or components.size(1) != embedding_dim:
            raise ValueError(
                f"Frozen PCA components shape mismatch: expect ({feat_dim}, {embedding_dim}), "
                f"got {tuple(components.shape)} from {resolved_pca_path}"
            )

        return (
            FrozenPCALayer(mean=mean, components=components),
            f"frozen PCA project {feat_dim} -> {embedding_dim} from '{resolved_pca_path}'",
        )

    raise ValueError(f"Unsupported text_proj_mode: {proj_mode}")


class PointWiseFeedForward(nn.Module):

    def __init__(self, hidden_size: int, dropout_rate: int):
        super(PointWiseFeedForward, self).__init__()

        self.conv1 = nn.Conv1d(hidden_size, hidden_size, kernel_size=1)
        self.dropout1 = nn.Dropout(p=dropout_rate)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv1d(hidden_size, hidden_size, kernel_size=1)
        self.dropout2 = nn.Dropout(p=dropout_rate)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        # inputs: (B, S, D)
        outputs = self.dropout2(self.conv2(self.relu(
            self.dropout1(self.conv1(inputs.transpose(-1, -2)))
        ))) # -> (B, D, S)
        outputs = outputs.transpose(-1, -2) # -> (B, S, D)
        outputs += inputs
        return outputs


class SASRec(freerec.models.SeqRecArch):

    def __init__(
        self, dataset: freerec.data.datasets.RecDataSet,
        maxlen: int = 50, embedding_dim: int = 64,
        dropout_rate: float = 0.2, num_blocks: int = 1, num_heads: int = 2,
        tfile: str = "", tfile_base: str = "", tfile_low: str = "",
        text_proj_mode: str = "auto", text_proj_pca_path: str = "",
        residual_alpha_init: float = 1.0, learn_residual_alpha: bool = False,
        gumbel_tau: float = 0.0,
        trainable_text: bool = False,
        alpha_fuse: bool = False, alpha_fuse_null_dim: int = 64,
    ) -> None:
        super().__init__(dataset)

        self.gumbel_tau = gumbel_tau

        self.alpha_fuse = False
        self.use_residual_text = bool(learn_residual_alpha)
        self.use_text = bool(tfile) or self.use_residual_text or alpha_fuse

        if self.use_residual_text:
            if tfile or not tfile_base or not tfile_low:
                raise ValueError(
                    "Residual-alpha mode requires --tfile-base and --tfile-low, and should not be mixed with --tfile."
                )

            tpath_base = os.path.join(dataset.path, tfile_base)
            tpath_low = os.path.join(dataset.path, tfile_low)
            base_feats = _load_item_features(tpath_base)
            low_feats = _load_item_features(tpath_low)

            if base_feats.size(0) != self.Item.count or low_feats.size(0) != self.Item.count:
                raise ValueError(
                    "Textual feats rows != dataset item count: "
                    f"base={base_feats.size(0)}, low={low_feats.size(0)}, Item.count={self.Item.count}. "
                    "Please align pkl row order to dataset internal item ids."
                )
            if base_feats.shape != low_feats.shape:
                raise ValueError(
                    f"Residual-alpha mode expects same shape for base/low feats, got "
                    f"base={tuple(base_feats.shape)} vs low={tuple(low_feats.shape)}"
                )

            feat_dim = base_feats.size(1)
            self.register_buffer("base_item_feats", base_feats)
            self.register_buffer("low_item_feats", low_feats)
            self.residual_alpha = nn.Parameter(torch.tensor(float(residual_alpha_init), dtype=torch.float32))
            self.in_proj, proj_msg = _build_input_projection(
                feat_dim=feat_dim,
                embedding_dim=embedding_dim,
                dataset_path=dataset.path,
                feature_path=tpath_base,
                proj_mode=text_proj_mode,
                pca_path=text_proj_pca_path,
                allow_infer_pca_path=False,
            )

            with torch.no_grad():
                init_final = self.low_item_feats + self.residual_alpha * (self.base_item_feats - self.low_item_feats)
                init_delta = (init_final - self.base_item_feats).abs().max().item()

            infoLogger(
                f"[SASRec] Residual-alpha text mode enabled, "
                f"base='{tfile_base}', low='{tfile_low}', shape={tuple(base_feats.shape)}, "
                f"{proj_msg}, alpha_init={float(residual_alpha_init):.6f}, "
                f"max|E_final(alpha_init)-E_base|={init_delta:.6e}"
            )
        elif self.use_text:
            tpath = os.path.join(dataset.path, tfile)
            text_feats = _load_item_features(tpath)

            if text_feats.size(0) != self.Item.count:
                raise ValueError(
                    f"Textual feats rows != dataset item count: feats={text_feats.size(0)} vs Item.count={self.Item.count}. "
                    f"This will cause index out of range. Please align pkl row order to dataset internal item ids."
                )

            feat_dim = text_feats.size(1)
            self.trainable_text = trainable_text
            if self.trainable_text:
                self.text_item_feats = nn.Parameter(text_feats)
            else:
                self.register_buffer("text_item_feats", text_feats)
            self.in_proj, proj_msg = _build_input_projection(
                feat_dim=feat_dim,
                embedding_dim=embedding_dim,
                dataset_path=dataset.path,
                feature_path=tpath,
                proj_mode=text_proj_mode,
                pca_path=text_proj_pca_path,
            )

            infoLogger(
                f"[SASRec] Loaded textual embeddings from '{tfile}', "
                f"shape={text_feats.shape}, trainable_text={trainable_text}, {proj_msg}"
            )

            if alpha_fuse:
                self.alpha_fuse = True
                self.alpha_fuse_null_dim = alpha_fuse_null_dim
                self.null_ID = nn.Embedding(
                    self.Item.count + self.NUM_PADS,
                    alpha_fuse_null_dim,
                    padding_idx=self.PADDING_VALUE
                )
                nn.init.zeros_(self.null_ID.weight)
                infoLogger(
                    f"[SASRec] AlphaFuse mode enabled, null_dim={alpha_fuse_null_dim}"
                )
            else:
                self.alpha_fuse = False
        else:
            # Original learnable ID embeddings
            self.Item.add_module(
                'embeddings', nn.Embedding(
                    num_embeddings=self.Item.count + self.NUM_PADS,
                    embedding_dim=embedding_dim,
                    padding_idx=self.PADDING_VALUE
                )
            )
            # ID embedding 已经是 embedding_dim，无需投影
            self.in_proj = nn.Identity()

        self.embedding_dim = embedding_dim
        self.num_blocks = num_blocks

        self.Position = nn.Embedding(maxlen, embedding_dim)
        self.embdDropout = nn.Dropout(p=dropout_rate)
        self.register_buffer(
            "positions",
            torch.tensor(range(0, maxlen), dtype=torch.long).unsqueeze(0)
        )

        self.attnLNs = nn.ModuleList() # to be Q for self-attention
        self.attnLayers = nn.ModuleList()
        self.fwdLNs = nn.ModuleList()
        self.fwdLayers = nn.ModuleList()

        self.lastLN = nn.LayerNorm(embedding_dim, eps=1e-8)

        for _ in range(num_blocks):
            self.attnLNs.append(nn.LayerNorm(
                embedding_dim, eps=1e-8
            ))

            self.attnLayers.append(
                nn.MultiheadAttention(
                    embed_dim=embedding_dim,
                    num_heads=num_heads,
                    dropout=dropout_rate,
                    batch_first=True # !!!
                )
            )

            self.fwdLNs.append(nn.LayerNorm(
                embedding_dim, eps=1e-8
            ))

            self.fwdLayers.append(PointWiseFeedForward(
                embedding_dim, dropout_rate
            ))

        # False True  True ...
        # False False True ...
        # False False False ...
        # ....
        # True indices that the corresponding position is not allowed to attend !
        self.register_buffer(
            'attnMask',
            torch.ones((maxlen, maxlen), dtype=torch.bool).triu(diagonal=1)
        )

        if cfg.loss == 'BCE':
            self.criterion = freerec.criterions.BCELoss4Logits(reduction='mean')
        elif cfg.loss == 'BPR':
            self.criterion = freerec.criterions.BPRLoss(reduction='mean')
        elif cfg.loss == 'CE':
            self.criterion = freerec.criterions.CrossEntropy4Logits(reduction='mean')

        self.reset_parameters()

    def reset_parameters(self):
        """Initializes the module parameters."""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.)
            elif isinstance(m, nn.Embedding):
                # Skip frozen embeddings (textual)
                if not m.weight.requires_grad:
                    continue
                nn.init.xavier_normal_(m.weight)
            elif isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d)):
                nn.init.constant_(m.weight, 1.)
                nn.init.constant_(m.bias, 0.)

    def sure_trainpipe(self, maxlen: int, batch_size: int):
        return self.dataset.train().shuffled_seqs_source(
           maxlen=maxlen
        ).seq_train_yielding_pos_(
            start_idx_for_target=1, end_idx_for_input=-1
        ).seq_train_sampling_neg_(
            num_negatives=1
        ).add_(
            offset=self.NUM_PADS, modified_fields=(self.ISeq,)
        ).lpad_(
            maxlen, modified_fields=(self.ISeq, self.IPos, self.INeg),
            padding_value=self.PADDING_VALUE
        ).batch_(batch_size).tensor_()

    def mark_position(self, seqs: torch.Tensor):
        positions = self.Position(self.positions) # (1, maxlen, D)
        return seqs + positions

    def current_text_item_feats(self) -> torch.Tensor:
        if self.use_residual_text:
            return self.low_item_feats + self.residual_alpha * (self.base_item_feats - self.low_item_feats)
        if self.use_text:
            return self.text_item_feats
        raise RuntimeError("current_text_item_feats is only available in textual modes")

    def lookup_text_item_feats(self, indices: torch.Tensor) -> torch.Tensor:
        feats = self.current_text_item_feats()
        padding_mask = indices == self.PADDING_VALUE
        safe_indices = (indices - self.NUM_PADS).clamp_min(0)
        gathered = feats[safe_indices]
        return gathered.masked_fill(padding_mask.unsqueeze(-1), 0.)

    def after_one_block(self, seqs: torch.Tensor, padding_mask: torch.Tensor, l: int):
        # inputs: (B, S, D)
        Q = self.attnLNs[l](seqs)
        seqs = self.attnLayers[l](
            Q, seqs, seqs,
            attn_mask=self.attnMask,
            need_weights=False
        )[0] + seqs

        seqs = self.fwdLNs[l](seqs)
        seqs = self.fwdLayers[l](seqs)

        return seqs.masked_fill(padding_mask, 0.)

    def encode(
        self, data: Dict[freerec.data.fields.Field, torch.Tensor]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        seqs = data[self.ISeq]
        padding_mask = (seqs == self.PADDING_VALUE).unsqueeze(-1)

        if self.alpha_fuse:
            base_feats = self.lookup_text_item_feats(seqs)  # (B,S,D)
            null_seqs = self.null_ID(seqs)  # (B,S,null_dim)
            feats = base_feats.clone()
            feats[..., -self.alpha_fuse_null_dim:] = base_feats[..., -self.alpha_fuse_null_dim:] + null_seqs
            seqs = feats

            all_base = self.current_text_item_feats()  # (N,D)
            all_null = self.null_ID.weight[self.NUM_PADS:]  # (N,null_dim)
            all_feats = all_base.clone()
            all_feats[..., -self.alpha_fuse_null_dim:] = all_base[..., -self.alpha_fuse_null_dim:] + all_null
            itemEmbds = self.in_proj(all_feats)
        elif self.use_text:
            seqs = self.lookup_text_item_feats(seqs) # 文本模式下: (B,S,768)
            itemEmbds = self.in_proj(self.current_text_item_feats())
        else:
            seqs = self.Item.embeddings(seqs)
            itemEmbds = self.Item.embeddings.weight[self.NUM_PADS:]

        seqs = self.in_proj(seqs)  # -> (B,S,64)
        seqs *= self.embedding_dim ** 0.5
        seqs = self.embdDropout(self.mark_position(seqs))
        seqs = seqs.masked_fill(padding_mask, 0.)

        for l in range(self.num_blocks):
            seqs = self.after_one_block(seqs, padding_mask, l)

        userEmbds = self.lastLN(seqs)  # (B,S,64)

        return userEmbds, itemEmbds

    def fit(
        self, data: Dict[freerec.data.fields.Field, torch.Tensor]
    ) -> Union[torch.Tensor, Tuple[torch.Tensor]]:
        userEmbds, itemEmbds = self.encode(data)
        indices = data[self.ISeq] != self.PADDING_VALUE
        userEmbds = userEmbds[indices] # (M, D)

        if cfg.loss in ('BCE', 'BPR'):
            posEmbds = itemEmbds[data[self.IPos][indices]] # (M, D)
            negEmbds = itemEmbds[data[self.INeg][indices]] # (M, D)
            posLogits = torch.einsum("MD,MD->M", userEmbds, posEmbds) # (M,)
            negLogits = torch.einsum("MD,MD->M", userEmbds, negEmbds) # (M,)

            if cfg.loss == 'BCE':
                if self.gumbel_tau > 0:
                    g_pos = -self.gumbel_tau * torch.log(-torch.log(torch.rand_like(posLogits).clamp_min(1e-8)) + 1e-8)
                    g_neg = -self.gumbel_tau * torch.log(-torch.log(torch.rand_like(negLogits).clamp_min(1e-8)) + 1e-8)
                    posLogits = posLogits + g_pos
                    negLogits = negLogits + g_neg
                posLabels = torch.ones_like(posLogits)
                negLabels = torch.zeros_like(negLogits)
                rec_loss = self.criterion(posLogits, posLabels) + \
                    self.criterion(negLogits, negLabels)
            elif cfg.loss == 'BPR':
                rec_loss = self.criterion(posLogits, negLogits)
        elif cfg.loss == 'CE':
            logits = torch.einsum("MD,ND->MN", userEmbds, itemEmbds) # (M, N)
            labels = data[self.IPos][indices] # (M,)
            rec_loss = self.criterion(logits, labels)

        return rec_loss

    def recommend_from_full(
        self, data: Dict[freerec.data.fields.Field, torch.Tensor]
    ) -> torch.Tensor:
        userEmbds, itemEmbds = self.encode(data)
        userEmbds = userEmbds[:, -1, :] # (B, D)
        return torch.einsum("BD,ND->BN", userEmbds, itemEmbds)

    def recommend_from_pool(
        self, data: Dict[freerec.data.fields.Field, torch.Tensor]
    ) -> torch.Tensor:
        userEmbds, itemEmbds = self.encode(data)
        userEmbds = userEmbds[:, -1, :] # (B, D)
        itemEmbds = itemEmbds[data[self.IUnseen]] # (B, K, D)
        return torch.einsum("BD,BKD->BK", userEmbds, itemEmbds)


class CoachForSASRec(freerec.launcher.Coach):

    def train_per_epoch(self, epoch: int):
        for data in self.dataloader:
            data = self.dict_to_device(data)
            loss = self.model(data)

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            self.monitor(
                loss.item(),
                n=len(data[self.User]), reduction="mean",
                mode='train', pool=['LOSS']
            )

    @torch.no_grad()
    def test_bucketed(self, bucket_mapping, ks=[5, 10]):
        self.model.eval()
        self.mode = 'test'

        buckets = {"head": {}, "mid": {}, "tail": {}}
        for b in buckets:
            for k in ks:
                buckets[b][f"HR@{k}"] = 0.0
                buckets[b][f"NDCG@{k}"] = 0.0
            buckets[b]["n"] = 0

        overall = {f"HR@{k}": 0.0 for k in ks}
        overall.update({f"NDCG@{k}": 0.0 for k in ks})
        overall["n"] = 0

        testpipe = self.model.sure_testpipe(self.cfg.maxlen, ranking=self.cfg.ranking)

        for data in testpipe:
            data = self.dict_to_device(data)
            bsz = data[self.Size] if self.Size in data else data[next(iter(data.keys()))].size(0)

            if self.cfg.ranking == 'pool':
                scores = self.model(data, ranking='pool')
            else:
                scores = self.model(data, ranking='full')
                if self.remove_seen:
                    seen = self.Item.to_csr(data[self.ISeen]).to(self.device).to_dense().bool()
                    scores[seen] = -1e23

            candidate_ids = data[self.IUnseen]

            for i in range(bsz):
                if self.cfg.ranking == 'pool':
                    pos_item_id = candidate_ids[i, 0].item()
                    pos_score = scores[i, 0].item()
                    neg_scores = scores[i, 1:]
                else:
                    pos_item_id = candidate_ids[i, 0].item()
                    pos_score = scores[i, 0].item()
                    neg_scores = scores[i, 1:]

                rank = (neg_scores > pos_score).sum().item()

                bucket = bucket_mapping.get(pos_item_id, "mid")

                for k in ks:
                    hr = 1.0 if rank < k else 0.0
                    ndcg = 1.0 / np.log2(rank + 2) if rank < k else 0.0
                    buckets[bucket][f"HR@{k}"] += hr
                    buckets[bucket][f"NDCG@{k}"] += ndcg
                    overall[f"HR@{k}"] += hr
                    overall[f"NDCG@{k}"] += ndcg

                buckets[bucket]["n"] += 1
                overall["n"] += 1

        infoLogger("\n" + "=" * 60)
        infoLogger("Per-Bucket Test Results")
        infoLogger("=" * 60)
        header = f"{'Bucket':>6s}  {'N':>6s}"
        for k in ks:
            header += f"  {'HR@'+str(k):>8s}  {'NDCG@'+str(k):>8s}"
        infoLogger(header)
        infoLogger("-" * 60)

        for bname in ["head", "mid", "tail"]:
            b = buckets[bname]
            n = b["n"]
            if n == 0:
                continue
            line = f"{bname:>6s}  {n:>6d}"
            for k in ks:
                line += f"  {b[f'HR@{k}']/n:>8.4f}  {b[f'NDCG@{k}']/n:>8.4f}"
            infoLogger(line)

        infoLogger("-" * 60)
        line = f"{'ALL':>6s}  {overall['n']:>6d}"
        for k in ks:
            line += f"  {overall[f'HR@{k}']/overall['n']:>8.4f}  {overall[f'NDCG@{k}']/overall['n']:>8.4f}"
        infoLogger(line)
        infoLogger("=" * 60)

        return buckets, overall


def main():

    dataset: freerec.data.datasets.RecDataSet
    try:
        dataset = getattr(freerec.data.datasets, cfg.dataset)(root=cfg.root)
    except AttributeError:
        dataset = freerec.data.datasets.RecDataSet(cfg.root, cfg.dataset, tasktag=cfg.tasktag)

    # ---- Compute or load item popularity buckets ----
    bucket_mapping = {}
    if cfg.bucket_file:
        if os.path.exists(cfg.bucket_file):
            raw = json.load(open(cfg.bucket_file))["mapping"]
            bucket_mapping = {int(k): v for k, v in raw.items()}
            infoLogger(f"[Buckets] Loaded {len(bucket_mapping)} item->bucket mappings from {cfg.bucket_file}")
        else:
            # Auto-compute from training data
            infoLogger("[Buckets] Computing item popularity buckets from training data...")
            item_counts = Counter()
            item_key = None
            for row in dataset.train().ordered_inter_source():
                if item_key is None:
                    item_key = [k for k in row if 'ITEM' in str(k) and 'ID' in str(k)][0]
                item_counts[row[item_key]] += 1
            items_by_freq = sorted(item_counts.items(), key=lambda x: -x[1])
            n = len(items_by_freq)
            n_head, n_tail = int(n * 0.2), int(n * 0.2)
            mapping = {}
            for i, (item, cnt) in enumerate(items_by_freq):
                if i < n_head:
                    mapping[int(item)] = "head"
                elif i >= n - n_tail:
                    mapping[int(item)] = "tail"
                else:
                    mapping[int(item)] = "mid"
            bucket_mapping = mapping
            os.makedirs(os.path.dirname(cfg.bucket_file) if os.path.dirname(cfg.bucket_file) else ".", exist_ok=True)
            json.dump({"mapping": mapping}, open(cfg.bucket_file, "w"))
            infoLogger(f"[Buckets] Saved {len(mapping)} mappings: head={n_head} mid={n-n_head-n_tail} tail={n_tail}")

    model = SASRec(
        dataset, maxlen=cfg.maxlen,
        embedding_dim=cfg.embedding_dim, dropout_rate=cfg.dropout_rate,
        num_blocks=cfg.num_blocks, num_heads=cfg.num_heads,
        tfile=cfg.tfile, tfile_base=cfg.tfile_base, tfile_low=cfg.tfile_low,
        text_proj_mode=cfg.text_proj_mode, text_proj_pca_path=cfg.text_proj_pca_path,
        residual_alpha_init=cfg.residual_alpha_init, learn_residual_alpha=cfg.learn_residual_alpha,
        gumbel_tau=cfg.gumbel_tau,
        trainable_text=cfg.trainable_text,
        alpha_fuse=cfg.alpha_fuse, alpha_fuse_null_dim=cfg.alpha_fuse_null_dim,
    )

    trainpipe = model.sure_trainpipe(cfg.maxlen, cfg.batch_size)
    validpipe = model.sure_validpipe(cfg.maxlen, ranking=cfg.ranking)
    testpipe = model.sure_testpipe(cfg.maxlen, ranking=cfg.ranking)

    coach = CoachForSASRec(
        dataset=dataset,
        trainpipe=trainpipe,
        validpipe=validpipe,
        testpipe=testpipe,
        model=model,
        cfg=cfg
    )

    infoLogger("\n" + "="*20 + " Model Architecture " + "="*20)
    infoLogger(model)
    infoLogger("="*60 + "\n")

    if cfg.test_only:
        infoLogger("[Test-Only] Loading checkpoint and running evaluation...")
        if cfg.ckpt_path:
            ckpt = torch.load(cfg.ckpt_path, map_location='cpu')
            model.load_state_dict(ckpt, strict=False)
            infoLogger(f"[Test-Only] Loaded state_dict from {cfg.ckpt_path}")
        else:
            coach.load_best()
        model.eval()
        coach.test(cfg.epochs)
        if bucket_mapping:
            coach.test_bucketed(bucket_mapping)
    else:
        coach.fit()
        if bucket_mapping:
            coach.load_best()
            coach.test_bucketed(bucket_mapping)


if __name__ == "__main__":
    main()
