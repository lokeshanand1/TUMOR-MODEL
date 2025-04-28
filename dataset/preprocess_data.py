import os
import cv2
import numpy as np
from sklearn.model_selection import train_test_split

# Paths
input_dir = "superSet"
output_dir = "dataset"

# Tumor categories
categories = ["meningioma", "glioma", "pituitary"]

# Create output directories
for category in categories:
    os.makedirs(f"{output_dir}/train/{category}", exist_ok=True)
    os.makedirs(f"{output_dir}/validation/{category}", exist_ok=True)

# Preprocessing function
def preprocess_image(image_path):
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)  # Load in grayscale
    image = (image - np.min(image)) / (np.max(image) - np.min(image))  # Normalize
    image_resized = cv2.resize(image, (256, 256))
    return image_resized

# Process and split images
for category in categories:
    category_path = os.path.join(input_dir, category)
    images = [f for f in os.listdir(category_path) if f.lower().endswith((".jpeg", ".jpg", ".png"))]
    train_imgs, val_imgs = train_test_split(images, test_size=0.2, random_state=42)

    for phase, file_list in [("train", train_imgs), ("validation", val_imgs)]:
        for filename in file_list:
            img_path = os.path.join(category_path, filename)
            processed_img = preprocess_image(img_path)
            save_path = f"{output_dir}/{phase}/{category}/{filename}"
            cv2.imwrite(save_path, (processed_img * 255).astype(np.uint8))
            print(f"Saved: {save_path}")
