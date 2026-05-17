# 5-fold split lists

One image filename per line (relative to `img/`), for example:

```
case_00.nii.gz
```

Layout:

```
splits/
├── abd/
│   ├── train_files_fold0.txt
│   └── val_files_fold0.txt
├── word/
└── amos/
```

Generate all folds for the toy example:

```bash
python dataset/split.py --create-splits-only --num-folds 5
```
