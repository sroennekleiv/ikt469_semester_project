import torch
import numpy as np

from .data_preprocess import PreProcessingClass
from keras.datasets import fashion_mnist
from torch.utils.data import Dataset, DataLoader, TensorDataset

class SplitDataset(Dataset):
    def __init__(self, split_fraction=0.5):
        self.split_fraction = split_fraction
        
    def split(self, X, y):
        assert len(X) == len(y), "Features and labels must have the same length"

        num_samples = len(X)
    
        rng = np.random.default_rng(42)
        indices = np.random.permutation(num_samples) # Shuffle indices
        rng.shuffle(indices)

        split_size = int(num_samples * self.split_fraction)

        split_indices = indices[:split_size]
        remaining_indices = indices[split_size:]

        X_split = X[split_indices]
        y_split = y[split_indices]

        X_remaining = X[remaining_indices]
        y_remaining = y[remaining_indices]

        return X_split, y_split, X_remaining, y_remaining
    
class FashionMNISTDataset:
    def __init__(self, batch_size=64, size=28, split_fraction=0.5):
        self.batch_size = batch_size
        self.size = size
        self.split_fraction = split_fraction

        (self.train_X, self.train_y), (self.test_X, self.test_y) = fashion_mnist.load_data()
        
        self.preprocessor = PreProcessingClass(size=self.size)
        self.splitter = SplitDataset(split_fraction=self.split_fraction)

    def get_dataloaders(self):
        # Split the train set into a smaller test set and a validation set
        self.train_X, self.train_y, self.val_X, self.val_y = self.splitter.split(self.train_X, self.train_y)

        # Preprocess the data
        train_X = torch.stack([
            self.preprocessor.preprocess(img, augment=True) for img in self.train_X
        ])

        val_X = torch.stack([
            self.preprocessor.preprocess(img, augment=False) for img in self.val_X
        ])

        test_X = torch.stack([
            self.preprocessor.preprocess(img, augment=False) for img in self.test_X
        ])

        # Convert labels to tensors
        train_y = torch.tensor(self.train_y, dtype=torch.long)
        val_y = torch.tensor(self.val_y, dtype=torch.long)
        test_y = torch.tensor(self.test_y, dtype=torch.long)

        # Create PyTorch datasets and dataloaders
        train_dataset = TensorDataset(train_X, train_y)
        val_dataset = TensorDataset(val_X, val_y)
        test_dataset = TensorDataset(test_X, test_y)

        self.train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)
        self.val_loader = DataLoader(val_dataset, batch_size=self.batch_size, shuffle=False)
        self.test_loader = DataLoader(test_dataset, batch_size=self.batch_size, shuffle=False)

        return self.train_loader, self.val_loader, self.test_loader
    
    def get_class_names(self):
        return ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat', 'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']
        
