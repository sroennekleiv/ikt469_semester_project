import torch
import torch.nn as nn
import torch.nn.functional as F

from .residual_block import ResidualNet
from .inception_block import InceptionNet
from .fire_module import SqueezeNet
    
# Router to dynamically assign input samples to different experts based on learned routing weights
class Router(nn.Module):
    def __init__(self, in_channels, num_experts):
        super().__init__()

        # Network to compute routing weights
        self.network = nn.Sequential(
            nn.Conv2d(in_channels, 16, 3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(16, num_experts)
        )
    
    def forward(self, x):
        return F.softmax(self.network(x), dim=-1)

class Experts(nn.Module):
    def __init__(self, experts, router):
        super().__init__()

        # List of expert models to process input samples based on routing decisions
        self.experts = nn.ModuleList(experts)
        self.router = router
        self.num_experts = len(experts)

    def forward(self, x):
        # Compute routing weights for each input sample
        routing_weights = self.router(x)

        # Normalize routing weights to ensure they sum to 1 across experts
        gate_weights = F.softmax(routing_weights, dim=-1)

        # Compute outputs from each expert model
        experts_outputs = [expert(x) for expert in self.experts]
        experts_outputs = torch.stack(experts_outputs, dim=1)

        # Combine expert outputs using the computed gate weights to produce final output
        gate_weights = gate_weights.unsqueeze(-1)
        output = (gate_weights * experts_outputs).sum(dim=1)

        return output
    
class MixtureOfExperts(nn.Module):
    def __init__(self, num_classes):
        super().__init__()

        # Initial convolutional layer to reduce spatial dimensions and increase channels
        self.stem = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )

        # Define the expert models to be used in the mixture of experts framework
        self.fire = SqueezeNet(num_classes)
        self.inception = InceptionNet(num_classes)
        self.residual = ResidualNet(num_classes)

        # Classifier to produce final predictions after combining expert outputs
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(256, num_classes)
        )

        self.projector = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(128, 64)
        )

        self.router = Router(32, 3)

        # Mixture of experts module to dynamically route input samples to different expert models based on learned routing weights
        self.experts = Experts([self.fire, self.inception, self.residual], self.router)

    def forward(self, x):
        x = self.stem(x)
        output = self.experts(x)
        return x, output