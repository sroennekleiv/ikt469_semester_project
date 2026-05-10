import torch
import torch.nn as nn
import torch.nn.functional as F

from .residualconvblock import ResiudualConvBlock

class AutoencoderModel(nn.Module):
    def __init__(self, in_channels=1, embedding_dim=128):
        super().__init__()
        
        self.encoder = nn.Sequential(
            ResiudualConvBlock(in_channels, 32, stride=2),
            ResiudualConvBlock(32, 64, stride=2),
            ResiudualConvBlock(64, 128, stride=2)
        )

        # Pooling layer to reduce spatial dimensions and capture global features to learn more compact and informative representations
        self.pool = nn.AdaptiveAvgPool2d(1)

        self.embedding_layer = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128, embedding_dim),
            nn.BatchNorm1d(embedding_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2)
        )

        self.decoder_input = nn.Sequential(
            nn.Linear(embedding_dim, 128 * 4 * 4),
            nn.ReLU(inplace=True)
        )

        self.decoder = nn.Sequential(
            nn.Unflatten(1, (128, 4, 4)),
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),  # 4x4 -> 8x8
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),  # 8x8 -> 16x16
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(32, 1, kernel_size=4, stride=2, padding=1),  # 16x16 -> 32x32
            nn.Sigmoid(),
        )

    def encode(self, x):
        features = self.encoder(x)
        pooled = self.pool(features)
        embedding = self.embedding_layer(pooled)
        embedding = F.normalize(embedding, dim=-1)
        return embedding
    
    def decode(self, x):
        decoded = self.decoder_input(x)
        reconstuct = self.decoder(decoded)
        return reconstuct

    def forward(self, x):
        embedding = self.encode(x)
        reconstruction = self.decode(embedding)

        return {
            "reconstruction": reconstruction,
            "embedding": embedding
        }