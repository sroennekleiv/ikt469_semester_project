import torch
import torch.nn as nn
import torch.nn.functional as F

from .fire_module import FireExpert
from .inception_block import InceptionExpert
from .residual_block import ResidualExpert
    
# Router module to compute routing weights for each expert based on the input features, 
# allowing the model to learn which expert to use for each input sample
class Router(nn.Module):
    def __init__(self, in_channels, num_experts):
        super().__init__()

        self.network = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),

            nn.Conv2d(64, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),

            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),

            nn.Linear(32, 64),
            nn.ReLU(inplace=True),

            nn.Dropout(0.2),

            nn.Linear(64, num_experts)
        )

        self.temperature = 2.0 # Temperature for optimized expert specialization scaling

    def forward(self, x):
        logits = self.network(x)
        weights = F.softmax(logits / self.temperature, dim=-1)
        return weights

# Experts module to process input samples based on routing decisions, allowing the model to learn specialized representations for different types of inputs
class Experts(nn.Module):
    def __init__(self, experts):
        super().__init__()

        # Initialize the experts as a ModuleList to allow for easy iteration and integration into the forward pass
        self.experts = nn.ModuleList(experts)

    def forward(self, x):
        outputs = [expert(x) for expert in self.experts]
        outputs = torch.stack(outputs, dim=1)

        return outputs

# Mixture of Experts model that combines the router and experts to produce final predictions, allowing for dynamic routing of inputs to specialized experts based on learned routing weights
class MixtureOfExperts(nn.Module):
    def __init__(self, num_classes, embedding_dim=128, projection_dim=64):
        super().__init__()

        # Initial convolutional layer to reduce spatial dimensions and increase channels
        self.stem = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2)
        )

        self.router = Router(32, num_experts=3) # Compute the routing weights for each expert based on input features

        # Define the expert models to be used in the mixture of experts framework
        self.experts = Experts([
            FireExpert(32, embedding_dim),
            InceptionExpert(32, embedding_dim),
            ResidualExpert(32, embedding_dim)
        ])

        # Classifier to produce final predictions after combining expert outputs
        self.classifier = nn.Sequential(
            nn.Linear(embedding_dim, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(128, num_classes)
        )

        # Projection head for the model to learn more discriminative embeddings for improved performance on downstream tasks
        self.projection_head = nn.Sequential(
            nn.Linear(embedding_dim, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, projection_dim)
        )

    def forward(self, x, return_routing=True):
        x = self.stem(x) # Extract features

        gate_weights = self.router(x) # Compute routing weights for each expert

        expert_embeddings = self.experts(x) # Get the embeddings from each expert

        # Compute fused embedding to combine the specialized representations from each expert
        fused_embedding = (gate_weights.unsqueeze(-1) * expert_embeddings).sum(dim=1)

        projection = F.normalize(self.projection_head(fused_embedding), dim=-1)

        logits = self.classifier(fused_embedding)

        if return_routing:
            return {
                "embedding": fused_embedding,
                "projection": projection,
                "logits": logits,
                "routing_weights": gate_weights,
                "expert_embeddings": expert_embeddings
            }

        return {
            "embedding": fused_embedding,
            "projection": projection,
            "logits": logits
        }
