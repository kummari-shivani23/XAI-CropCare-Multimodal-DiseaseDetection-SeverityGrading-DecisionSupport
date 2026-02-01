import os
import shutil
import random

RAW_DIR = "data/raw/grapes"
OUT_DIR = "data/processed/grapes"

SPLITS = {
    "train": 0.7,
    "val": 0.15,
    "test": 0.15
}

random.seed(42)

for cls in os.listdir(RAW_DIR):
    cls_path = os.path.join(RAW_DIR, cls)
    images = os.listdir(cls_path)
    random.shuffle(images)

    total = len(images)
    train_end = int(SPLITS["train"] * total)
    val_end = train_end + int(SPLITS["val"] * total)

    split_files = {
        "train": images[:train_end],
        "val": images[train_end:val_end],
        "test": images[val_end:]
    }

    for split, files in split_files.items():
        split_dir = os.path.join(OUT_DIR, split, cls)
        os.makedirs(split_dir, exist_ok=True)

        for img in files:
            src = os.path.join(cls_path, img)
            dst = os.path.join(split_dir, img)
            shutil.copy(src, dst)

print("✅ Dataset split completed successfully.")
