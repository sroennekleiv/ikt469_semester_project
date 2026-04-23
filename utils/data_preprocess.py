import torch
from torchvision.transforms import transforms

class PreProcessingClass:
    def __init__(self, size=32):
        self.size = size

    def preprocess(self, image, augment=False):
        if isinstance(image, torch.Tensor):
            image = image.numpy()

        if augment: # Apply data augmentation during training
            self.transform = transforms.Compose([
                transforms.ToPILImage(),
                transforms.Resize((self.size, self.size)),
                transforms.RandomResizedCrop(self.size, scale=(0.8, 1.0)),
                transforms.RandomRotation(10),
                transforms.ToTensor(),
                transforms.Normalize((0.5,), (0.5,))
            ])
        else:
            self.transform = transforms.Compose([
                transforms.ToPILImage(),
                transforms.Resize((self.size, self.size)),
                transforms.ToTensor(),
                transforms.Normalize((0.5,), (0.5,))
            ])
        
        return self.transform(image)
        