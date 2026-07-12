from torchvision.transforms import Compose, RandomCrop, Resize, GaussianBlur, ColorJitter, ToTensor

# As stated in the paper the transofrmations are:
# 1. Random Crop
# 2. Resize
# 3. Color Distortion
# 4. Gaussian Blur

image_augmentation = Compose([
    Resize(size=224),
    RandomCrop(size=180),
    Resize(size=224),
    ColorJitter(
        brightness=0.4,
        contrast=0.4,
        saturation=0.4,
        hue=0.4,
    ),
    GaussianBlur(kernel_size=21),
    ToTensor()
])
