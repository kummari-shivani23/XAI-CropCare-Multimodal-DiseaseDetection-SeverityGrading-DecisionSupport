from image_index import get_image_list
import pandas as pd

image_root = "data/processed/grapes/train"
image_list = get_image_list(image_root)

df = pd.DataFrame({"image_path": image_list})
df.to_csv("data/metadata/image_index.csv", index=False)

images = pd.read_csv("data/metadata/image_index.csv")
severity = pd.read_csv("data/metadata/grape_severity.csv")

merged = pd.concat([images, severity], axis=1)
merged.to_csv("data/metadata/image_severity_map.csv", index=False)

print("✅ Image–severity alignment completed")
