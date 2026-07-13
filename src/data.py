from datasets import load_dataset
from src.augmentation import image_augmentation
from torch.utils.data import IterableDataset

class CustomImageNetDataset(IterableDataset):

    def __init__(self,test_size: int = 1000):
         super().__init__()
         dataset = load_dataset("evanarlian/imagenet_1k_resized_256", streaming=True, split="train").shuffle(seed=42, buffer_size=100)
         self.train_dataset = dataset.skip(test_size)
         self.test_dataset = dataset.take(test_size)

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
