from __future__ import annotations
import argparse
import json
import shutil
from pathlib import Path
import numpy as np
try:
    import nibabel as nib
except ImportError:
    nib = None
from dataset.paths import (
    DATA_ROOT,
    DATASET_NAMES,
    PROJECT_ROOT,
    SPLITS_ROOT,
    read_split_list,
    split_list_path,
)
TOY_SHAPE = (64, 96, 96)
ORGAN_LABEL = {
    "abd": 3,
    "word": [3, 4],
    "amos": 6,
}
CASE_NAMES = ("case_00.nii.gz", "case_01.nii.gz")
def _write_nifti(path: Path, array: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if nib is None:
        raise RuntimeError("nibabel is required. Install with: pip install nibabel")
    nib.save(nib.Nifti1Image(array.astype(np.float32), np.eye(4)), str(path))
def _make_image(rng: np.random.Generator) -> np.ndarray:
    """Pseudo-CT volume in Hounsfield-like range."""
    base = rng.normal(loc=-50.0, scale=80.0, size=TOY_SHAPE).astype(np.float32)
    d, h, w = TOY_SHAPE
    base[d // 4 : 3 * d // 4, h // 4 : 3 * h // 4, w // 4 : 3 * w // 4] += 120.0
    return base
def _make_label(rng: np.random.Generator, organ) -> np.ndarray:
    label = np.zeros(TOY_SHAPE, dtype=np.float32)
    d, h, w = TOY_SHAPE
    label[d // 3 : 2 * d // 3, h // 3 : 2 * h // 3, w // 3 : 2 * w // 3] = 1.0
    if isinstance(organ, list):
        pass
    else:
        label[label > 0] = float(organ)
    return label
def create_toy_volumes(seed: int = 42) -> None:
    rng = np.random.default_rng(seed)
    for name in DATASET_NAMES:
        img_dir = DATA_ROOT / name / "img"
        lbl_dir = DATA_ROOT / name / "label"
        for case in CASE_NAMES:
            _write_nifti(img_dir / case, _make_image(rng))
            _write_nifti(lbl_dir / case, _make_label(rng, ORGAN_LABEL[name]))
    aug_root = DATA_ROOT / "abd2"
    for sub in ("img", "label"):
        src = DATA_ROOT / "abd" / sub
        dst = aug_root / sub
        if dst.exists():
            shutil.rmtree(dst)
        shutil.copytree(src, dst)
    manifest = {
        "layout": "toy",
        "shape_dhw": TOY_SHAPE,
        "cases": list(CASE_NAMES),
        "datasets": list(DATASET_NAMES),
        "augmented_abd": "abd2",
    }
    (DATA_ROOT / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"Toy volumes written under: {DATA_ROOT}")
def create_split_files(num_folds: int = 5) -> None:
    """One train / one val case per dataset per fold (toy-friendly)."""
    for fold in range(num_folds):
        for ds in DATASET_NAMES:
            for split, case in zip(("train", "val"), CASE_NAMES):
                out = split_list_path(ds, fold, split)
                out.parent.mkdir(parents=True, exist_ok=True)
                out.write_text(f"{case}\n", encoding="utf-8")
    print(f"Split lists written under: {SPLITS_ROOT}")
def verify_toy() -> bool:
    ok = True
    for ds in DATASET_NAMES:
        for split in ("train", "val"):
            p = split_list_path(ds, 0, split)
            if not p.is_file():
                print(f"MISSING split file: {p}")
                ok = False
                continue
            names = read_split_list(p)
            for case in names:
                img = DATA_ROOT / ds / "img" / case
                lbl = DATA_ROOT / ds / "label" / case
                if not img.is_file() or not lbl.is_file():
                    print(f"MISSING volume: {img} or {lbl}")
                    ok = False
    aug = DATA_ROOT / "abd2" / "img" / CASE_NAMES[0]
    if not aug.is_file():
        print(f"MISSING augmented abdomen copy: {aug}")
        ok = False
    return ok
def main() -> None:
    parser = argparse.ArgumentParser(description="Toy data & split utilities for DiffResmem")
    parser.add_argument("--create-toy", action="store_true", help="Write synthetic NIfTI + split lists")
    parser.add_argument("--create-splits-only", action="store_true", help="Only write split .txt files")
    parser.add_argument("--verify", action="store_true", help="Check toy files exist")
    parser.add_argument("--num-folds", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    if args.verify:
        raise SystemExit(0 if verify_toy() else 1)
    if args.create_toy:
        create_toy_volumes(seed=args.seed)
        create_split_files(num_folds=args.num_folds)
        if not verify_toy():
            raise SystemExit("Toy setup incomplete after generation.")
        print("Toy example is ready.")
        return
    if args.create_splits_only:
        create_split_files(num_folds=args.num_folds)
        return
    parser.print_help()
if __name__ == "__main__":
    main()
