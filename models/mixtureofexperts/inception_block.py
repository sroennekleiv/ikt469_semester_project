import torch
import torch.nn as nn

# Inception block to capture multi-scale features by applying parallel convolutions with different kernel sizes
class InceptionBlock(nn.Module):
    def __init__(self, in_channels, channels_1x1, channels_3x3, channels_5x5, pool_channels):
        super().__init__()

        # 1x1 convolution branch to capture fine-grained features
        self.branch1 = nn.Sequential(
            nn.Conv2d(in_channels, channels_1x1, 1),
            nn.ReLU()
        )

        # 3x3 convolution branch to capture medium-scale features
        self.branch2 = nn.Sequential(
            nn.Conv2d(in_channels, channels_3x3, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(channels_3x3, channels_3x3, 3, padding=1),
            nn.ReLU()
        )

        # 5x5 convolution branch to capture larger-scale features
        self.branch3 = nn.Sequential(
            nn.Conv2d(in_channels, channels_5x5, 5, padding=2),
            nn.ReLU(),
            nn.Conv2d(channels_5x5, channels_5x5, 5, padding=2),
            nn.ReLU()
        )

        # Pooling branch to capture spatial context and reduce dimensions
        self.branch4 = nn.Sequential(
            nn.MaxPool2d(3, stride=1, padding=1),
            nn.Conv2d(in_channels, pool_channels, 1),
            nn.ReLU()
        )
    
    def forward(self, x):
        # Compute outputs from each branch
        b1 = self.branch1(x)
        b2 = self.branch2(x)
        b3 = self.branch3(x)
        b4 = self.branch4(x)

        # Concatenate outputs along the channel dimension to combine multi-scale features
        return torch.cat([b1, b2, b3, b4], dim=1)
    
class InceptionNet(nn.Module):
    def __init__(self, num_classes):
        super().__init__()

        # Initial convolutional layer to reduce spatial dimensions and increase channels
        self.stem = nn.Sequential(
            nn.Conv2d(32, 32, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )

        # Inception blocks to learn multi-scale features
        self.inception_blocks = nn.Sequential(
            InceptionBlock(32, 16, 16, 16, 16),
            InceptionBlock(64, 32, 32, 32, 32),
            InceptionBlock(128, 64, 64, 64, 64),
        )

        # Classifier to produce final predictions
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(256, num_classes)
        )
    
    def forward(self, x):
        x = self.stem(x)
        x = self.inception_blocks(x)
        x = self.classifier(x)
        return x