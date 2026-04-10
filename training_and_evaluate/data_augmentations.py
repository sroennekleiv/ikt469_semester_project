from torchvision.transforms import transforms

train_transforms = transforms.Compose([
    transforms.toPILImage(),
    transforms.RandomResizedCrop(32, scale=(0.8, 1.0)),
    transforms.RandomHorizontalFlip(),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
    transforms.RandomGrayscale(p=0.1),
    transforms.ToTensor(),
])

test_transforms = transforms.Compose([
    transforms.toPILImage(),
    transforms.ToTensor(),
])