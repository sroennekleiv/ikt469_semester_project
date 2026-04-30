import torch
import torch.nn as nn
import torch.nn.functional as F

class ExpertTrainerAndEvaluator:
    def __init__(self, model, device):
        self.model = model.to(device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=1e-3)
        self.scheduler = torch.optim.lr_scheduler.StepLR(
            self.optimizer, step_size=15, gamma=0.5
        )
        self.criterion = nn.CrossEntropyLoss()
        self.device = device
        self.load_balance_weight = 0.01

    def load_balancing_loss(self, routing_weights, num_experts=3):
        # Sum routing probabilities across batch to get expert importance
        expert_importance = routing_weights.sum(dim=0)  # Shape: [num_experts]

        # Normalize to probabilities
        expert_importance = expert_importance / expert_importance.sum()

        # Uniform distribution (all experts equally likely)
        uniform = torch.ones(num_experts, device=self.device) / num_experts

        # KL divergence to encourage balanced routing
        kl_loss = F.kl_div(
            F.log_softmax(expert_importance, dim=0),
            uniform,
            reduction='batchmean'
        )
        return kl_loss

    def train(self, train_loader):
        self.model.train()
        total_loss, correct, total = 0, 0, 0

        for x, y in train_loader:
            x, y = x.to(self.device), y.to(self.device)

            self.optimizer.zero_grad()

            # Get outputs AND routing weights for load balancing
            _, outputs, routing_weights = self.model(x, return_routing=True)

            # Classification loss
            ce_loss = self.criterion(outputs, y)

            # Load balancing loss (prevents gating collapse)
            lb_loss = self.load_balancing_loss(routing_weights, num_experts=3)

            # Combined loss: 99% classification + 1% load balancing
            loss = ce_loss + self.load_balance_weight * lb_loss

            loss.backward()
            self.optimizer.step()

            total_loss += loss.item() * x.size(0)
            _, predicted = outputs.max(1)
            correct += predicted.eq(y).sum().item()
            total += y.size(0)

        avg_loss = total_loss / total
        accuracy = correct / total
        return avg_loss, accuracy
    
    def evaluate(self, test_loader):
        self.model.eval()
        total_loss, correct, total = 0, 0, 0

        with torch.no_grad():
            for x, y in test_loader:
                x, y = x.to(self.device), y.to(self.device)

                # Only get outputs (not routing weights) during evaluation
                _, outputs = self.model(x, return_routing=False)
                loss = self.criterion(outputs, y)

                total_loss += loss.item() * x.size(0)
                _, predicted = outputs.max(1)
                correct += predicted.eq(y).sum().item()
                total += y.size(0)

        avg_loss = total_loss / total
        accuracy = correct / total
        return avg_loss, accuracy
    
    def extract_embeddings(self, data_loader):
        self.model.eval()
        embeddings = []
        labels = []

        with torch.no_grad():
            for x, y in data_loader:
                x = x.to(self.device)
                features = self.model.feature_extractor(x)
                pooled = self.model.pooling(features)
                embeddings.append(pooled.cpu())
                labels.append(y)
        
        return torch.cat(embeddings, dim=0), torch.stack(labels)
    
    def run_experiment(self, train_loader, test_loader, epochs):
        for epoch in range(epochs):
            train_loss, train_acc = self.train(train_loader)
            self.scheduler.step()  # Update learning rate
            print(f'Epoch {epoch+1}/{epochs} - Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}')

        test_loss, test_acc = self.evaluate(test_loader)
        print(f'Test Loss: {test_loss:.4f}, Test Acc: {test_acc:.4f}')

        return test_loss, test_acc