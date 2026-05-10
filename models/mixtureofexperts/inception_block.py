import torch
import torch.nn as nn

# Inception block to capture multi-scale features by applying parallel convolutions with different kernel sizes
class InceptionBlock(nn.Module):
    def __init__(self, in_channels, channels_1x1, channels_3x3, channels_5x5, pool_channels):
        super().__init__()

        # 1x1 convolution branch to capture fine-grained features
        self.branch1 = nn.Sequential(
            nn.Conv2d(in_channels, channels_1x1, kernel_size=1),
            nn.BatchNorm2d(channels_1x1),
            nn.ReLU(inplace=True)
        )

        # 3x3 convolution branch to capture medium-scale features
        self.branch2 = nn.Sequential(
            nn.Conv2d(in_channels, channels_3x3, kernel_size=1),
            nn.BatchNorm2d(channels_3x3),
            nn.ReLU(inplace=True),

            nn.Conv2d(channels_3x3, channels_3x3, kernel_size=3, padding=1),
            nn.BatchNorm2d(channels_3x3),
            nn.ReLU(inplace=True)
        )

        # 5x5 convolution branch to capture larger-scale features
        self.branch3 = nn.Sequential(
            nn.Conv2d(in_channels, channels_5x5, kernel_size=1),
            nn.BatchNorm2d(channels_5x5),
            nn.ReLU(inplace=True),

            nn.Conv2d(channels_5x5, channels_5x5, kernel_size=5, padding=2),
            nn.BatchNorm2d(channels_5x5),
            nn.ReLU(inplace=True)
        )

        # Pooling branch to capture spatial context and reduce dimensions
        self.branch4 = nn.Sequential(
            nn.MaxPool2d(kernel_size=3, stride=1, padding=1),
            nn.Conv2d(in_channels, pool_channels, kernel_size=1),
            nn.BatchNorm2d(pool_channels),
            nn.ReLU(inplace=True)
        )
    
    def forward(self, x):
        # Compute outputs from each branch
        b1 = self.branch1(x)
        b2 = self.branch2(x)
        b3 = self.branch3(x)
        b4 = self.branch4(x)

        # Concatenate outputs along the channel dimension to combine multi-scale features
        return torch.cat([b1, b2, b3, b4], dim=1)
    
class InceptionExpert(nn.Module):
    def __init__(self, in_channels=32, embedding_dim=128):
        super().__init__()

        # Inception blocks to learn multi-scale features
        self.layers = nn.Sequential(
            InceptionBlock(in_channels, 16, 16, 16, 16),
            InceptionBlock(64, 32, 32, 32, 32),
            InceptionBlock(128, 64, 64, 64, 64),
        )

        # Classifier to produce final predictions
        self.embedding = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(256, embedding_dim),
            nn.BatchNorm1d(embedding_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2)
        )
    
    def forward(self, x):
        x = self.layers(x)
        embedding = self.embedding(x)
        return embedding