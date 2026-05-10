import random
import torch
import torch.nn as nn
import torch.nn.functional as F

from .residualconvblock import ResiudualConvBlock

class ContrastiveModel(nn.Module):
    def __init__(self, in_channels=1, embedding_dim=128, projection_dim=64, num_classes=10):
        super().__init__()

        self.encoder = nn.Sequential(
            ResiudualConvBlock(in_channels, 64, stride=2),
            ResiudualConvBlock(64, 128, stride=2),
            ResiudualConvBlock(128, 256, stride=2),
            ResiudualConvBlock(256, 256, stride=2),
        )

        self.pool = nn.AdaptiveAvgPool2d(1)
        
        self.embedding_head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256, embedding_dim),
            nn.BatchNorm1d(embedding_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2)
        )

        self.projector = nn.Sequential(
            nn.Linear(embedding_dim, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Linear(256, projection_dim)
        )

        self.classifier = nn.Sequential(
            nn.Linear(embedding_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, num_classes)
        )
    
    def forward(self, x):
        # Extract features using the encoder and flatten them
        features = self.encoder(x)
        pooled = self.pool(features)

        # Project features to the embedding space and normalize
        embedding = self.embedding_head(pooled)
        embedding = F.normalize(embedding, dim=1)

        proj = self.projector(embedding)
        proj = F.normalize(proj, dim=1)

        logits = self.classifier(embedding)
        return {
            "embedding": embedding,
            "projection": proj,
            "logits": logits
        }