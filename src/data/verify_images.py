from PIL import Image
import os

def verify_images(folder):
    bad_images = []
    for root, _, files in os.walk(folder):
        for file in files:
            try:
                img_path = os.path.join(root, file)
                with Image.open(img_path) as img:
                    img.verify()
            except:
                bad_images.append(img_path)
    return bad_images

bad = verify_images("data/raw/grapes")
print("Corrupted images:", bad)
