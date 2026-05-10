import torch
import torch.nn as nn
import torch.nn.functional as F

class FireModule(nn.Module):
    def __init__(self, in_channels, squeeze_channels, expand_channels):
        super().__init__()

        # Squeeze layer to reduce the number of channels and parameters to learn more compact representations
        self.squeeze = nn.Sequential(
            nn.Conv2d(in_channels, squeeze_channels, kernel_size=1),
            nn.BatchNorm2d(squeeze_channels),
            nn.ReLU(inplace=True)
        )

        # Expapand layer with 1x1 convolutions to capture fine-grained features and maintain computational efficiency
        self.expand1x1 = nn.Sequential(
            nn.Conv2d(squeeze_channels, expand_channels, kernel_size=1),
            nn.BatchNorm2d(expand_channels),
            nn.ReLU(inplace=True)
        )

        # Expand layer with 3x3 convolutions to capture more complex features and multi-scale information
        self.expand3x3 = nn.Sequential(
            nn.Conv2d(
                squeeze_channels,
                expand_channels,
                kernel_size=3,
                padding=1
            ),
            nn.BatchNorm2d(expand_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.squeeze(x) # Learn compact representations

        x1 = self.expand1x1(x) # Capture fine-grained features
        x3 = self.expand3x3(x) # Capture more complex features and multi-scale information

        return torch.cat([x1, x3], dim=1)
    
class FireExpert(nn.Module):
    def __init__(self, in_channels=32, embedding_dim=128):
        super().__init__()

        # Stack of Fire modules to learn hierarchical feature representations
        self.layers = nn.Sequential(
            FireModule(in_channels, 16, 32),
            FireModule(64, 16, 64),
            FireModule(128, 32, 128),
        )

        # Embedding layer to produce a fixed-size representation for each expert
        self.embedding = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(256, embedding_dim),
            nn.BatchNorm1d(embedding_dim),
            nn.ReLU(),
            nn.Dropout(0.2)
        )
    
    def forward(self, x):
        x = self.layers(x)
        embedding = self.embedding(x)
        return embedding