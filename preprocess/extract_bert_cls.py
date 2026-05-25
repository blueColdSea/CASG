import argparse
import os
import pickle

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
from transformers import AutoConfig, AutoModel, AutoTokenizer


class BertClsExtractor:
    def __init__(self, model_name: str, device: str, max_length: int):
        self.device = device
        self.max_length = max_length

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        config = AutoConfig.from_pretrained(model_name, output_hidden_states=True)
        self.model = AutoModel.from_pretrained(model_name, config=config).to(self.device).eval()
        for param in self.model.parameters():
            param.requires_grad = False
        self.hidden_size = config.hidden_size

    def encode_tsv(self, text_file: str, batch_size: int) -> np.ndarray:
        df = pd.read_csv(text_file, sep="\t", dtype={"ITEM": int, "TEXT": str})
        df["TEXT"] = df["TEXT"].fillna("").astype(str)
        item_ids = df["ITEM"].astype(int).tolist()
        texts = df["TEXT"].tolist()

        num_items = max(item_ids) + 1
        outputs = np.zeros((num_items, self.hidden_size), dtype=np.float32)

        total_batches = (len(texts) + batch_size - 1) // batch_size
        for batch_idx in tqdm(range(total_batches), desc=f"[CLS] {os.path.basename(text_file)}"):
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
                last_hidden_state = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    token_type_ids=token_type_ids,
                ).last_hidden_state
                cls_vectors = last_hidden_state[:, 0, :].cpu().numpy().astype(np.float32)

            for item_id, vector in zip(item_ids[start:end], cls_vectors):
                outputs[item_id] = vector

        return outputs


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--text-file", required=True, help="Input TSV with ITEM and TEXT columns")
    parser.add_argument("--out-file", required=True, help="Output PKL path")
    parser.add_argument("--model-name", default="bert-base-uncased")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--max-length", type=int, default=64)
    args = parser.parse_args()

    extractor = BertClsExtractor(
        model_name=args.model_name,
        device=args.device,
        max_length=args.max_length,
    )
    features = extractor.encode_tsv(args.text_file, batch_size=args.batch_size)

    os.makedirs(os.path.dirname(args.out_file), exist_ok=True)
    with open(args.out_file, "wb") as file_obj:
        pickle.dump(features, file_obj)

    print(f"[OK] Saved: {args.out_file}")
    print(f"[Info] shape: {features.shape}")


if __name__ == "__main__":
    main()
