import torch
from transformers import AutoProcessor, GitVisionModel
from PIL import Image
import os
from tqdm import tqdm

# 1) Device setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 2) Load GIT’s vision-only encoder
processor    = AutoProcessor.from_pretrained("microsoft/git-large-coco")
vision_model = GitVisionModel.from_pretrained("microsoft/git-large-coco")\
                            .to(device).eval()

def extract_features(image_paths):
    """Extract [257,1024] vision tokens for each image in image_paths."""
    feats = []
    for img_path in tqdm(image_paths, desc="Extracting GIT features"):
        img    = Image.open(img_path).convert("RGB")
        inputs = processor(images=img, return_tensors="pt")\
                 .pixel_values.to(device)           # -> [1,3,224,224]
        with torch.no_grad():
            out  = vision_model(pixel_values=inputs)
            feat = out.last_hidden_state.squeeze(0).cpu()  # -> [257,1024]
        feats.append(feat)
    return torch.stack(feats)  # -> [N,257,1024]

def get_sorted_image_paths(root_dir, multi=True):
    """
    Walk root_dir/<class_folder> in sorted order.
    If multi=True, grab all images in each folder (training).
    If multi=False, grab only the first sorted image (test).
    """
    paths = []
    for cls in sorted(os.listdir(root_dir)):
        cls_dir = os.path.join(root_dir, cls)
        if not os.path.isdir(cls_dir):
            continue
        imgs = sorted(
            f for f in os.listdir(cls_dir)
            if f.lower().endswith((".jpg", ".png", ".jpeg"))
        )
        if not imgs:
            continue
        if multi:
            for img in imgs:
                paths.append(os.path.join(cls_dir, img))
        else:
            # only one test image per class
            paths.append(os.path.join(cls_dir, imgs[0]))
    return paths

if __name__ == "__main__":
    # 3) Update these to your actual data roots:
    train_root = "/DATA/deep_learning/eeg-to-img/data/osfstorage-archive/training_images"
    test_root  = "/DATA/deep_learning/eeg-to-img/data/osfstorage-archive/test_images"

    # 4) Build sorted lists
    train_paths = get_sorted_image_paths(train_root, multi=True)
    test_paths  = get_sorted_image_paths(test_root,  multi=False)

    # 5) Extract & save train features
    train_feats = extract_features(train_paths)  # [N_train,257,1024]
    torch.save({"img_features": train_feats},
               "/DATA/deep_learning/eeg-to-img/data/weights/EEG_Image_decode/ViT-L-14_features_GIT_train.pt")
    print("Saved TRAIN features:", train_feats.shape)

    # 6) Extract & save test features
    test_feats = extract_features(test_paths)    # [N_test,257,1024]
    torch.save({"img_features": test_feats},
               "/DATA/deep_learning/eeg-to-img/data/weights/EEG_Image_decode/ViT-L-14_features_GIT_test.pt")
    print("Saved  TEST features:", test_feats.shape)
