import torch
import torch.nn as nn
import torch.nn.functional as F

import open_clip

class PretrainedCLIPModel(nn.Module):
    def __init__(self, model_arc='ViT-B-32', pretrained='openai', embedding_dim=512, projection_dim=128, num_classes=10, freeze=True, device='cuda'):
        super().__init__()
        self.device = device

        # Load the pretrained CLIP model and its corresponding preprocessing transforms
        self.model, _, self.preprocess = open_clip.create_model_and_transforms(model_arc, pretrained)
        self.model = self.model.to(device)

        if freeze:
            for param in self.model.parameters():
                param.requires_grad = False

        self.embedding_dim = embedding_dim

        self.projection_head = nn.Sequential(
            nn.Linear(self.embedding_dim, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(256, projection_dim)
        )

        self.classifier = nn.Sequential(
            nn.Linear(embedding_dim, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(256, num_classes)
        )

    def encode_func(self, x):
        # Convert the grayscale images to RGB colors
        if x.shape[1] == 1:
            x = x.repeat(1, 3, 1, 1)

        # Preprocess the images using the CLIP preprocessing transforms
        x = F.interpolate(
            x,
            size=(224, 224),
            mode="bilinear",
            align_corners=False
        )

        image_features = self.model.encode_image(x) # Extract image features using the CLIP model's image encoder

        image_features = F.normalize(
            image_features,
            dim=-1
        )

        return image_features

    def project_func(self, x):
        projection = self.projection_head(x)
        projection = F.normalize(projection, dim=-1)
        return projection
    
    def classification_func(self, x):
        logits = self.classifier(x)
        return logits
    
    def forward(self, x):
        embedding = self.encode_func(x)
        proj = self.project_func(embedding)
        logits = self.classification_func(embedding)

        return {
            "embedding": embedding,
            "projection": proj,
            "logits": logits
        }
