import torch
import torch.nn as nn
import torch.nn.functional as F

class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()

        # Convolutional layers to learn residual features
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, stride=stride, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1, stride=1)
        self.bn2 = nn.BatchNorm2d(out_channels)

        # Skip the connection to allow the block to learn identity mappings if needed
        self.skip = nn.Identity()

        # If the input and output dimensions differ, use a convolutional layer to match them for the skip connection
        if stride != 1 or in_channels != out_channels:
            self.skip = nn.Sequential(
                nn.Conv2d(
                    in_channels,
                    out_channels,
                    kernel_size=1,
                    stride=stride
                ),
                nn.BatchNorm2d(out_channels)
            )
    
    def forward(self, x):
        # Compute the residual features through the convolutional layers
        residual = F.relu(self.bn1(self.conv1(x)))
        residual = self.bn2(self.conv2(residual))

        # Add the skip connection to the residual features and apply activation
        skip = self.skip(x)
        output = F.relu(residual + skip) 
        return output
    
class ResidualExpert(nn.Module):
    def __init__(self, in_channels=32, embedding_dim=128):
        super().__init__()

        # Residual blocks to learn complex features while maintaining gradient flow
        self.layers = nn.Sequential(
            ResidualBlock(in_channels, 64, stride=2),
            ResidualBlock(64, 128, stride=2),
            ResidualBlock(128, 256, stride=2)
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