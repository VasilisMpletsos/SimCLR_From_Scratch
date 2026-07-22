from datasets import load_dataset
from torch.utils.data import IterableDataset

from src.augmentation import image_augmentation


class CustomImageNetDataset(IterableDataset):
    def __init__(self, split):
        super().__init__()
        self.dataset = load_dataset(
            "evanarlian/imagenet_1k_resized_256", streaming=True, split=split
        ).shuffle(seed=42, buffer_size=100)

    def __iter__(self):
        for sample in self.dataset:
            image = sample["image"]
            # label = sample['label']
            x_i = image_augmentation(image)
            x_j = image_augmentation(image)

            yield x_i, x_j
