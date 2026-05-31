import os
import shutil
import logging
from pathlib import Path
from typing import List, Tuple

import cv2

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Supported image extensions
IMG_EXTS = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff'}

def is_image_file(path: Path) -> bool:
    return path.suffix.lower() in IMG_EXTS

def validate_image(path: Path) -> bool:
    """Return True if image can be opened by OpenCV.
    Corrupt files will cause cv2.imread to return None.
    """
    try:
        img = cv2.imread(str(path))
        return img is not None
    except Exception as e:
        logger.debug(f"Failed to read image {path}: {e}")
        return False

def scan_dataset(root_dir: Path) -> Tuple[List[Path], List[Path]]:
    """Walk the root_dir and return a tuple of (valid_files, invalid_files)."""
    valid = []
    invalid = []
    for dirpath, _, filenames in os.walk(root_dir):
        for fname in filenames:
            fpath = Path(dirpath) / fname
            if not is_image_file(fpath):
                invalid.append(fpath)
                continue
            if validate_image(fpath):
                valid.append(fpath)
            else:
                invalid.append(fpath)
    return valid, invalid

def split_dataset(
    files: List[Path],
    output_dir: Path,
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
    test_ratio: float = 0.1,
    seed: int = 42,
) -> None:
    """Split files into train/val/test preserving class subfolders.
    Assumes that the immediate parent of each image is the class label.
    """
    import random

    random.seed(seed)
    # Group by class label (parent folder name)
    class_to_files = {}
    for f in files:
        label = f.parent.name
        class_to_files.setdefault(label, []).append(f)

    for label, items in class_to_files.items():
        random.shuffle(items)
        n = len(items)
        n_train = int(n * train_ratio)
        n_val = int(n * val_ratio)
        splits = {
            'train': items[:n_train],
            'val': items[n_train:n_train + n_val],
            'test': items[n_train + n_val:]
        }
        for split_name, split_files in splits.items():
            target_dir = output_dir / split_name / label
            target_dir.mkdir(parents=True, exist_ok=True)
            for src in split_files:
                shutil.copy2(src, target_dir / src.name)

def log_invalid_files(invalid_files: List[Path], log_path: Path) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with open(log_path, "w", encoding="utf-8") as f:
        for p in invalid_files:
            f.write(f"{p}\n")
    logger.info(f"Logged {len(invalid_files)} invalid files to {log_path}")

def run_ingestion(
    raw_dir: str = "data/raw",
    processed_dir: str = "data/processed",
    log_path: str = "data/processed/ingestion_log.txt",
) -> None:
    raw_path = Path(raw_dir)
    processed_path = Path(processed_dir)
    logger.info(f"Scanning raw dataset at {raw_path}")
    valid, invalid = scan_dataset(raw_path)
    logger.info(f"Found {len(valid)} valid images, {len(invalid)} invalid/corrupt files")
    log_invalid_files(invalid, Path(log_path))
    split_dataset(valid, processed_path)
    logger.info(f"Dataset split into train/val/test under {processed_path}")

if __name__ == "__main__":
    run_ingestion()
