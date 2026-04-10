import random
import torch
import torch.nn as nn
import torch.nn.functional as F

class ContrastiveModel(nn.Module):
    def __init__(self, in_channels=3, embedding_dim=128, projection_dim=64):
        super().__init__()

        self.backbone = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),

            nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),

            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),

            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),

            nn.Conv2d(256, 256, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),

            nn.AdaptiveAvgPool2d((1, 1))
        )

        self.embedding = nn.Linear(256, embedding_dim)

        self.projector = nn.Sequential(
            nn.Linear(embedding_dim, embedding_dim),
            nn.ReLU(),
            nn.Linear(embedding_dim, projection_dim)
        )
    
    def forward(self, x):
        # Extract features using the backbone and flatten them
        features = self.backbone(x)
        features = torch.flatten(features, start_dim=1)

        # Project features to the embedding space and normalize
        z = self.embedding(features)
        z = F.normalize(z, dim=1)

        p = self.projector(z)
        p = F.normalize(p, dim=1)
        return z, p
    
    def get_label_pairs(self, x, y):
        pairs = []
        targets = []

        for i in range(len(y)):
            x1, y1 = x[i], y[i]

            positive_indices = (y == y1).nonzero(as_tuple=True)[0]
            positive_indices = positive_indices[positive_indices != i] # Exclude the current sample

            if len(positive_indices) > 0:
                positive = positive_indices[torch.randint(0, len(positive_indices), (1,)).item()]
                pairs.append((x1, x[positive]))
                targets.append(1.0)

            negative_indices = (y != y1).nonzero(as_tuple=True)[0]
            negative_indices = negative_indices[negative_indices != i] # Exclude the current sample

            if len(negative_indices) > 0:
                negative = negative_indices[torch.randint(0, len(negative_indices), (1,)).item()]
                pairs.append((x1, x[negative]))
                targets.append(-1.0)

        return pairs, torch.tensor(targets, dtype=torch.float32)