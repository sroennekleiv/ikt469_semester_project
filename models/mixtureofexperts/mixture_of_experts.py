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

        # IMPROVED: Deeper router network with more capacity
        self.network = nn.Sequential(
            nn.Conv2d(in_channels, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(32, 64),
            nn.ReLU(),
            nn.Linear(64, num_experts)
        )

    def forward(self, x):
        logits = self.network(x)
        # Add temperature scaling for better specialization
        temperature = 5.0
        return F.softmax(logits / temperature, dim=-1)

class Experts(nn.Module):
    def __init__(self, experts, router):
        super().__init__()

        # List of expert models to process input samples based on routing decisions
        self.experts = nn.ModuleList(experts)
        self.router = router
        self.num_experts = len(experts)

    def forward(self, x, return_routing=False):
        # Compute routing weights for each input sample
        gate_weights = self.router(x)  # Already softmax from router

        # Compute outputs from each expert model
        experts_outputs = [expert(x) for expert in self.experts]
        experts_outputs = torch.stack(experts_outputs, dim=1)

        # Combine expert outputs using the computed gate weights to produce final output
        gate_weights_expanded = gate_weights.unsqueeze(-1)
        output = (gate_weights_expanded * experts_outputs).sum(dim=1)

        if return_routing:
            return output, gate_weights
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

    def forward(self, x, return_routing=False):
        x = self.stem(x)
        if return_routing:
            output, routing_weights = self.experts(x, return_routing=True)
            return x, output, routing_weights
        else:
            output = self.experts(x, return_routing=False)
            return x, output