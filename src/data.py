from datasets import load_dataset
from torch.utils.data import IterableDataset

from src.augmentation import image_augmentation


class CustomImageNetDataset(IterableDataset):
    def __init__(self):
        super().__init__()
        self.train_dataset = load_dataset(
            "evanarlian/imagenet_1k_resized_256", streaming=True, split="train"
        ).shuffle(seed=42, buffer_size=100)
        self.test_dataset = load_dataset(
            "evanarlian/imagenet_1k_resized_256", streaming=True, split="val"
        )

    def __iter__(self):
        for sample in self.train_dataset:
            image = sample["image"]
            # label = sample['label']
            x_i = image_augmentation(image)
            x_j = image_augmentation(image)

            yield x_i, x_j

    def get_test_data(self):
        for sample in self.test_dataset:
            image = sample["image"]
            x_i = image_augmentation(image)
            x_j = image_augmentation(image)

            yield x_i, x_j
