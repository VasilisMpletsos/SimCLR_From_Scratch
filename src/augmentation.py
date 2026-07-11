from torchvision.transforms import Compose, RandomCrop, Resize, GaussianBlur, ColorJitter

# As stated in the paper the transofrmations are:
# 1. Random Crop
# 2. Resize
# 3. Color Distortion
# 4. Gaussian Blur

image_augmentation = Compose([
    # TODO: Change the size of random cropping & original size
    RandomCrop(size=180),
    Resize(size=256),
    ColorJitter(
        brightness=0.4,
        contrast=0.4,
        saturation=0.4,
        hue=0.4,
    ),
    GaussianBlur(kernel_size=63)
])
