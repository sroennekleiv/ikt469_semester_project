import torch

from torch.utils.data import DataLoader
from torchvision import transforms
from datasets import load_dataset

class Cifar100Dataset:
    def __init__(self, batch_size=64, num_workers=2):
        self.batch_size = batch_size
        self.num_workers = num_workers

        self.dataset = load_dataset('uoft-cs/cifar100')

        self.train_transform = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(32, padding=4),
            transforms.ToTensor(),
            transforms.Normalize((0.5071, 0.4867, 0.4408),
                                 (0.2675, 0.2565, 0.2761))
        ])

        self.test_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5071, 0.4867, 0.4408),
                                 (0.2675, 0.2565, 0.2761))
        ])

        self.train_ds = self.dataset["train"].with_transform(self._train_transform)
        self.test_ds = self.dataset["test"].with_transform(self._test_transform)
    
    def _train_transform(self, batch):
        images = [self.train_transform(img.convert("RGB")) for img in batch["img"]]
        labels = batch["fine_label"]
        return {"image": images, "label": labels}

    def _test_transform(self, batch):
        images = [self.test_transform(img.convert("RGB")) for img in batch["img"]]
        labels = batch["fine_label"]
        return {"image": images, "label": labels}
    
    def get_dataloaders(self):
        train_loader = DataLoader(
            self.train_ds,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers
        )

        test_loader = DataLoader(
            self.test_ds,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers
        )

        return train_loader, test_loader

    def get_class_names(self):
        return self.dataset['train'].features['fine_label'].names