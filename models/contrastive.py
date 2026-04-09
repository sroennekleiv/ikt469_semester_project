import random
import torch
import torch.nn as nn
import torchvision.models as models

class ContrastiveModel(nn.Module):
    def __init__(self, embedding_dim=128, projection_dim=64):
        super().__init__()

        backbone = models.resnet18(weights=None)
        backbone.fc = nn.Identity()

        self.backbone = backbone

        self.embedding = nn.Linear(512, embedding_dim)
        self.projector = nn.Sequential(
            nn.Linear(embedding_dim, embedding_dim),
            nn.ReLU(),
            nn.Linear(embedding_dim, projection_dim)
        )
    
    def forward(self, x):
        h = self.backbone(x)
        z = self.embedding(h)
        p = self.projector(z)
        return z, p
    
    def get_label_pairs(self, x, y):
        pairs = []
        targets = []

        for i in range(len(x)):
            x1, y1 = x[i], y[i]

            positive = random.choice((y == y1).nonzero(as_tuple=True)[0])
            pairs.append((x1, x[positive]))
            targets.append(1)

            negative = random.choice((y != y1).nonzero(as_tuple=True)[0])
            pairs.append((x1, x[negative]))
            targets.append(0)

        return pairs, torch.tensor(targets)