# DiffResmem

Official code for *Towards Generic Abdominal Multi-Organ Segmentation with multiple partially labeled datasets* (CMIG, 2025).

Diffusion-based segmentation with sparse memory (`ResUnet_mem_sparse`). Paths are relative to the repo root or set via env vars (see below).

## Setup

```bash
python -m venv venv && source venv/bin/activate   # Windows: venv\Scripts\activate
pip install torch
pip install -r requirement.txt
```

## Toy smoke test

```bash
python dataset/split.py --create-toy
python train/5fold_DiffResmem.py --fold 0 --toy --max-epochs 2 --batch-size 1
python test/test_5fold_DiffResmem.py --fold 0 --toy --logdir ./checkpoints/5fold_DiffResmem0/model/<checkpoint>.pt
```

Toy data: `data/toy/{abd,word,amos,abd2}/{img,label}/`. Splits: `dataset/splits/{abd,word,amos}/*.txt` (one filename per line).

## Full training

```bash
export DIFFRESMEM_LAYOUT=paper
export DIFFRESMEM_DATA_ROOT=./data
python train/5fold_DiffResmem.py --fold 0
python test/test_5fold_DiffResmem.py --fold 0 --logdir <checkpoint.pt>
```

**Train data:** `data/respacing_{abd,word,amos,abd2}/{img,label}/`  
**Test data:** `data/processed_{abd,word,amos,abd2}/{img,label}/`  
**Splits:** `dataset/splits/<dataset>/train_files_fold{N}.txt`, `val_files_fold{N}.txt`

Regenerate split lists: `python dataset/split.py --create-splits-only`

## Environment variables

| Variable | Default | Purpose |
|----------|---------|---------|
| `DIFFRESMEM_LAYOUT` | `toy` | `toy` or `paper` |
| `DIFFRESMEM_DATA_ROOT` | `./data/toy` | Image root |
| `DIFFRESMEM_SPLITS_ROOT` | `./dataset/splits` | Split `.txt` root |

Run scripts from the repo root (or `PYTHONPATH=.`).

## Citation

Please cite the CMIG 2025 paper if you use this code.
