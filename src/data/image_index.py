import os

def get_image_list(image_root):
    images = []
    for cls in sorted(os.listdir(image_root)):
        cls_path = os.path.join(image_root, cls)
        if os.path.isdir(cls_path):
            for img in sorted(os.listdir(cls_path)):
                images.append(os.path.join(cls, img))
    return images
