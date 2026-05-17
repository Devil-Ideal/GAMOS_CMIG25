from __future__ import annotations
import os
from pathlib import Path
PROJECT_ROOT = Path(
    os.environ.get("DIFFRESMEM_PROJECT_ROOT", Path(__file__).resolve().parents[1])
).resolve()
LAYOUT = os.environ.get("DIFFRESMEM_LAYOUT", "toy").lower()
_DEFAULT_DATA = PROJECT_ROOT / "data" / ("toy" if LAYOUT == "toy" else "processed")
DATA_ROOT = Path(os.environ.get("DIFFRESMEM_DATA_ROOT", _DEFAULT_DATA)).resolve()
SPLITS_ROOT = Path(
    os.environ.get("DIFFRESMEM_SPLITS_ROOT", PROJECT_ROOT / "dataset" / "splits")
).resolve()
DATASET_NAMES = ("abd", "word", "amos")
SPLITS = ("train", "val")
def _subdir(name: str, layout: str) -> str:
    """Map logical dataset name to on-disk directory name."""
    if layout == "toy":
        return name
    if layout == "paper":
        mapping = {
            "abd_train": "respacing_abd",
            "word_train": "respacing_word",
            "amos_train": "respacing_amos",
            "abd_test": "processed_abd",
            "word_test": "processed_word",
            "amos_test": "processed_amos",
        }
        return mapping[name]
    raise ValueError(f"Unknown layout: {layout}")
def train_dataset_dir(dataset: str) -> Path:
    if LAYOUT == "toy":
        return DATA_ROOT / dataset
    return DATA_ROOT / _subdir(f"{dataset}_train", LAYOUT)
def test_dataset_dir(dataset: str) -> Path:
    if LAYOUT == "toy":
        return DATA_ROOT / dataset
    return DATA_ROOT / _subdir(f"{dataset}_test", LAYOUT)
def augmented_abd_dir() -> Path:
    """Second abdomen source (path must contain 'abd2' for dataset label routing)."""
    if LAYOUT == "toy":
        return DATA_ROOT / "abd2"
    return DATA_ROOT / "respacing_abd2"
def split_list_path(dataset: str, fold: int, split: str) -> Path:
    """e.g. dataset/splits/abd/train_files_fold0.txt"""
    return SPLITS_ROOT / dataset / f"{split}_files_fold{fold}.txt"
def read_split_list(path: Path) -> list[str]:
    with open(path, "r", encoding="utf-8") as f:
        return [line.strip() for line in f if line.strip()]
def image_paths_from_split(dataset: str, fold: int, split: str, *, for_test: bool) -> list[str]:
    rel_names = read_split_list(split_list_path(dataset, fold, split))
    root = test_dataset_dir(dataset) if for_test else train_dataset_dir(dataset)
    img_dir = root / "img"
    return [str(img_dir / name) for name in rel_names]
def resolve_abd_aug_paths(abd_paths: list[str]) -> list[str]:
    """Mirror abdomen list to the augmented abdomen folder."""
    aug_root = str(augmented_abd_dir())
    if LAYOUT == "toy":
        primary = str(train_dataset_dir("abd"))
        return [p.replace(primary, aug_root) for p in abd_paths]
    return [p.replace("respacing_abd", "respacing_abd2") for p in abd_paths]
def resolve_abd_aug_paths_processed(abd_paths: list[str]) -> list[str]:
    if LAYOUT == "toy":
        return resolve_abd_aug_paths(abd_paths)
    return [p.replace("processed_abd", "processed_abd2") for p in abd_paths]
