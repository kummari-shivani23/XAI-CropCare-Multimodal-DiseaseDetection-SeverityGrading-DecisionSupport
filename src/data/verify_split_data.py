import os

BASE = "data/processed/grapes"

for split in ["train", "val", "test"]:
    print(f"\n{split.upper()}")
    for cls in os.listdir(os.path.join(BASE, split)):
        count = len(os.listdir(os.path.join(BASE, split, cls)))
        print(f"{cls}: {count}")
