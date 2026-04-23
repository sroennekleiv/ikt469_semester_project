import torch
import numpy as np

from .data_preprocess import PreProcessingClass
from keras.datasets import fashion_mnist
from torch.utils.data import Dataset, DataLoader, TensorDataset

    
class FashionMNISTDataset:
    def __init__(self, batch_size=128, size=32, split_fraction=0.5):
        self.batch_size = batch_size
        self.size = size
        self.split_fraction = split_fraction

        (self.train_X, self.train_y), (self.test_X, self.test_y) = fashion_mnist.load_data()
        
        self.preprocessor = PreProcessingClass(size=self.size)

    def get_dataloaders(self, is_contrastive=False):
        if is_contrastive:
            train_X = torch.tensor(self.train_X, dtype=torch.uint8)
            test_X = torch.tensor(self.test_X, dtype=torch.uint8)
        else:
            # Preprocess the data
            train_X = torch.stack([
                self.preprocessor.preprocess(img, augment=True) for img in self.train_X
            ])

            test_X = torch.stack([
                self.preprocessor.preprocess(img, augment=False) for img in self.test_X
            ])

        #print(f"Shape: {train_X.shape}")

        # Convert labels to tensors
        train_y = torch.tensor(self.train_y, dtype=torch.long)
        test_y = torch.tensor(self.test_y, dtype=torch.long)

        # Create PyTorch datasets and dataloaders
        train_dataset = TensorDataset(train_X, train_y)
        test_dataset = TensorDataset(test_X, test_y)

        self.train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)
        self.test_loader = DataLoader(test_dataset, batch_size=self.batch_size, shuffle=False)

        return self.train_loader, self.test_loader
    
    def get_class_names(self):
        return ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat', 'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']
        
