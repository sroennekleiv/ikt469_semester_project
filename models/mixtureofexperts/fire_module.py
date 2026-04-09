import torch
import torch.nn as nn
import torch.nn.functional as F

class FireModuleModel(nn.Module):
    def __init__(self, in_channels, squeeze_channels, expand_channels):
        super().__init__()

        # Squeeze layer to reduce the number of channels
        self.squeeze = nn.Sequential(
            nn.Conv2d(in_channels, squeeze_channels, 1),
            nn.ReLU()
        )

        # Expand layers to learn multi-scale features
        self.expand1x1 = nn.Sequential(
            nn.Conv2d(squeeze_channels, expand_channels, 1),
            nn.ReLU()
        )

        # Expand layer with 3x3 convolutions to capture spatial context
        self.expand3x3 = nn.Sequential(
            nn.Conv2d(squeeze_channels, expand_channels, 3, padding=1),
            nn.ReLU()
        )

    def forward(self, x):
        x = F.relu(self.squeeze(x))
        x1 = F.relu(self.expand1x1(x))
        x3 = F.relu(self.expand3x3(x))
        return torch.cat([x1, x3], dim=1)
    
class SqueezeNet(nn.Module):
    def __init__(self, num_classes):
        super().__init__()

        # Stem layer to reduce spatial dimensions and increase channels
        self.stem = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=3),
            nn.ReLU(),
            nn.MaxPool2d(3, padding=1)
        )

        # Fire modules to learn multi-scale features and reduce parameters
        self.fire_modules = nn.Sequential(
            FireModuleModel(32, 16, 32),
            FireModuleModel(64, 16, 64),
            FireModuleModel(128, 32, 128),
        )

        # Classifier to produce final predictions
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(256, num_classes)
        )
    
    def forward(self, x):
        x = self.stem(x)
        x = self.fire_modules(x)
        x = self.classifier(x)
        return x