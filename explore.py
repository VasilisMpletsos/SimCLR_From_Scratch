# %% Imports
from src.augmentation import image_augmentation
import matplotlib.pyplot as plt
import seaborn as sns
from datasets import load_dataset
import torch
from torchvision.transforms.functional import pil_to_tensor

# %% Tests
ds = load_dataset("evanarlian/imagenet_1k_resized_256", streaming=True, split="train").shuffle(seed=42, buffer_size=1)


# %% Get images
image = next(iter(ds))['image']

# %% Get images
plt.imshow(image)

# %% Get images
ax = plt.figure(figsize=(15,10)).subplot_mosaic(
    """
    ABCDE
    """
);
ax['A'].imshow(image);
ax['A'].axis("off");
ax['A'].set_title("Original Image")

tensor_image = pil_to_tensor(image)

image1 = image_augmentation(tensor_image).permute(1,2,0)
ax['B'].imshow(image1);
ax['B'].axis("off");

image2 = image_augmentation(tensor_image).permute(1,2,0)
ax['C'].imshow(image2);
ax['C'].axis("off");

image3 = image_augmentation(tensor_image).permute(1,2,0)
ax['D'].imshow(image3);
ax['D'].axis("off");

image4 = image_augmentation(tensor_image).permute(1,2,0)
ax['E'].imshow(image4);
ax['E'].axis("off");
plt.tight_layout()
