# PolyFilter: Learning Topology-Aligned Semantic IDs for Sequential Recommendation

Code for reproducing the core experiments in the PolyFilter paper.

## Overview

PolyFilter learns topology-aligned semantic representations for sequential recommendation by applying learnable graph spectral filters (Bernstein polynomials) over an item-item co-occurrence graph.

The pipeline has three stages:

1. **Preprocess** — extract BERT [CLS] embeddings from item text
2. **Filter** — train Bernstein polynomial coefficients on the item-item graph
3. **SASRec** — train SASRec with the filtered embeddings

## Environment

Python 3.10+, PyTorch 2.0+, CUDA recommended.

```bash
pip install -r requirements.txt
```

The SASRec training uses [FreeRec](https://github.com/Re-bin/FreeRec) (≥0.8.7) as the training framework.

## Data Preparation

### Dataset format

Each dataset follows the FreeRec directory layout under a `data/` directory:

```
data/Processed/{dataset_name}/
├── train.txt       # user_id\titem_id_1\titem_id_2\t...\titem_id_n
├── valid.txt       # same format
├── test.txt        # same format
├── item.txt        # item_id: int (one per line)
└── schema.pkl      # FreeRec schema file
```

### Item text

A TSV file with two columns (ITEM and TEXT):

```
ITEM\tTEXT
0\t"Product description text..."
1\t"Another product description..."
...
```

For the three Amazon/Luxury/Musical datasets used in the paper, item text was built from concatenating title + category + description fields of the Amazon review metadata.

### Train interaction file

A TSV file for filter training (USER, ITEM, TIMESTAMP columns):

```
USER\tITEM\tTIMESTAMP
0\t42\t1375344000
0\t17\t1375345000
...
```

## Step-by-Step Reproduction

### Step 1: Extract BERT Embeddings

```bash
python preprocess/extract_bert_cls.py \
    --text-file /path/to/item_text.tsv \
    --out-file /path/to/output/bert_cls.pkl \
    --model-name bert-base-uncased \
    --max-length 64 \
    --batch-size 64 \
    --device cuda
```

Output: a `.pkl` file containing a `(num_items, 768)` float32 numpy array.

### Step 2: Train Bernstein Filter

```bash
python filter/train_bernstein.py \
    --text_file /path/to/item_text.tsv \
    --train_file /path/to/train_interactions.tsv \
    --output_dir /path/to/filter_output/ \
    --bern_k 2 \
    --alpha 0.001 \
    --max_length 64 \
    --topk_neighbors 20 \
    --epochs 100 \
    --lr 0.01 \
    --device cuda
```

Key hyperparameters:
- `--bern_k` — Bernstein polynomial order (K=2 for BERT, K=4 for Stella in the paper)
- `--alpha` — preservation loss weight (0.001 for BERT, 0 for Stella)
- `--topk_neighbors` — per-item neighbor truncation (20 in the paper)

Outputs in `--output_dir`:
- `id_emb_HGC_bern_k{bern_k}_alpha{alpha}_hidden768.pkl` — filtered embeddings
- `*_meta.json` — metadata (args, final loss)

### Step 3: Train SASRec

Place the filtered `.pkl` file under `data/Processed/{dataset_name}/`, then:

```bash
# Baseline (raw BERT)
python sasrec/main.py \
    --config=sasrec/configs/Amazon2018Luxury_440_LOU.yaml \

# PolyFilter (Bernstein-filtered)
python sasrec/main.py \
    --config=sasrec/configs/Amazon2018Luxury_440_LOU_bern.yaml \
```


### Config File Reference

The provided configs follow this structure:

```yaml
device: '0'          # GPU device ID
root: ../data        # path to data directory
dataset: ...         # dataset name (matches data/Processed/{name})
tfile: ...           # embedding pkl filename (relative to dataset dir)
text_proj_mode: ...  # projection mode (omit for identity, linear for filtered)
maxlen: 50
num_heads: 1
num_blocks: 2
embedding_dim: 64
dropout_rate: 0.5
epochs: 500
batch_size: 2048
optimizer: adam
lr: 5.e-4
weight_decay: 1.e-5
loss: BCE
tasktag: NEXTITEM
monitors: [LOSS, HitRate@1, HitRate@5, HitRate@10, NDCG@5, NDCG@10]
which4best: NDCG@5
```

