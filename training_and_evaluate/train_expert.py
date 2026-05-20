import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from models.mixtureofexperts.expert_specialization.expert_specialization_loss_functions import orthogonality_loss, routing_sharpness_loss, expert_consistency_loss, routing_entropy_loss, loading_balancing_loss

class ExpertTrainerAndEvaluator:
    def __init__(self, model, device, lr=1e-3, contrastive_weight=0.1, entropy_weight=0.05, load_balance_weight=0.1):
        self.model = model.to(device)
        self.device = device

        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr, weight_decay=1e-4)
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=50)
        self.criterion = nn.CrossEntropyLoss()

        self.contrastive_weight = contrastive_weight
        self.load_balance_weight = load_balance_weight

        self.orthogonality_weight = 0.1
        self.sharpness_weight = 0.05
        self.consistency_weight = 0.5
        self.entropy_weight = entropy_weight

        self.routing_history = []
        self.specialization_history = []

    # Function to compute the specialization score based on the entropy of the routing weights, 
    # where a lower mean entropy indicates stronger specialization of the experts
    def compute_specialization_score(self, routing_weights):
        # Get the routing weights as a numpy array
        routing_np = routing_weights.detach().cpu().numpy()

        # Compute the entropy of the routing distribution for each sample
        entropy = -np.sum(routing_np * np.log(routing_np + 1e-8), axis=1)

        mean_entropy = entropy.mean() # Get max value which indicate most balanced routing
        max_entropy = np.log(routing_np.shape[1])

        # Compute specialization score as 1 - (mean entropy / max entropy), 
        # where a higher score indicates stronger specialization of the experts
        specialization_score = (1 - mean_entropy / max_entropy)
        return specialization_score

    def train(self, train_loader):
        self.model.train()

        total_loss = 0
        total_correct = 0
        total_samples = 0

        total_ce_loss = 0
        total_orth_loss = 0
        total_sharpness_loss = 0
        total_consistency_loss = 0
        total_entropy_loss = 0
        total_load_balance_loss = 0

        for x, y in train_loader:
            x = x.to(self.device)
            y = y.to(self.device)

            self.optimizer.zero_grad()

            outputs_1 = self.model(x, return_routing=True)

            # Extract the logits
            logits = outputs_1["logits"]

            # Extract the routing weights and expert embeddings for computing the specialization losses
            routing_weights = outputs_1["routing_weights"]
            expert_embeddings = outputs_1["expert_embeddings"]

            # Compute the various loss components for expert specialization and routing effectiveness
            orth_loss = orthogonality_loss(expert_embeddings)
            sharpness_loss = routing_sharpness_loss(routing_weights)
            consistency_loss = expert_consistency_loss(expert_embeddings, routing_weights)
            entropy_loss = routing_entropy_loss(routing_weights)
            balance_loss = loading_balancing_loss(routing_weights)

            ce_loss = self.criterion(logits, y)

            # Combine all the loss components with their respective weighting factors to compute the total loss for backpropagation
            loss = (
                ce_loss
                #+ self.orthogonality_weight * orth_loss
                #+ self.sharpness_weight * sharpness_loss
                #+ self.consistency_weight * consistency_loss
                + self.entropy_weight * entropy_loss
                + self.load_balance_weight * balance_loss
            )

            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()
            total_loss += loss.item() * x.size(0)
            total_ce_loss += ce_loss.item()
            
            #total_orth_loss += orth_loss.item()
            #total_sharpness_loss += sharpness_loss.item()
            #total_consistency_loss += consistency_loss.item()

            total_entropy_loss += entropy_loss.item()
            total_load_balance_loss += balance_loss.item()

            predictions = logits.argmax(dim=1)
            total_correct += (predictions.eq(y).sum().item())

            total_samples += y.size(0)

            # Get the expert usage by averaging the routing weights across the batch
            if routing_weights.dim() == 3:
                expert_usage = routing_weights.mean(dim=(0, 1))
            else:
                expert_usage = routing_weights.mean(dim=0)
            self.routing_history.append(expert_usage.detach().cpu().numpy())

        average_loss = total_loss / total_samples
        accuracy = total_correct / total_samples

        print("\nTraining Statistics")
        print(f"CE Loss: {total_ce_loss / len(train_loader):.4f}")
        print(f"Load Balancing Loss: {total_load_balance_loss / len(train_loader):.4f}")
        print(f"Entropy Loss: {total_entropy_loss / len(train_loader):.4f}")
        #print(f"Orthogonality Loss: {total_orth_loss / len(train_loader):.4f}")
        #print(f"Sharpness Loss: {total_sharpness_loss / len(train_loader):.4f}")
        #print(f"Consistency Loss: {total_consistency_loss / len(train_loader):.4f}")

        return average_loss, accuracy
    
    def evaluate(self, test_loader):
        self.model.eval()

        total_loss = 0
        total_correct = 0
        total_samples = 0

        expert_usage_accumulator = 0

        with torch.no_grad():
            for x, y in test_loader:
                x = x.to(self.device)
                y = y.to(self.device)

                outputs = self.model(x, return_routing=True)

                logits = outputs["logits"]
                routing_weights = outputs["routing_weights"]

                # Compute loss
                loss = self.criterion(logits, y)
                total_loss += loss.item() * x.size(0)

                # Accuracy
                predictions = logits.argmax(dim=1)
                total_correct += predictions.eq(y).sum().item()
                total_samples += y.size(0)

                # Handle routing shape
                if routing_weights.dim() == 3:  # [batch, top_k, experts]
                    expert_usage = routing_weights.mean(dim=(0,1))
                else:  # [batch, experts]
                    expert_usage = routing_weights.mean(dim=0)

                # Weighted accumulation
                expert_usage_accumulator += expert_usage.detach() * x.size(0)

        average_expert_usage = expert_usage_accumulator / total_samples

        print("Evaluation Expert Usage")
        for i, usage in enumerate(average_expert_usage):
            print(f"Expert {i}: {usage.item():.4f}")

        average_loss = total_loss / total_samples
        accuracy = total_correct / total_samples

        return average_loss, accuracy

    
    def run_experiment(self, train_loader, test_loader, epochs):
        best_accuracy = 0.0

        for epoch in range(epochs):
            train_loss, train_acc = self.train(train_loader)
            self.scheduler.step()  # Update the learning rate based on the scheduler after each epoch
            print(f'\nEpoch {epoch+1}/{epochs} - Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}')

            test_loss, test_acc = self.evaluate(test_loader)
            print(f'Test Loss: {test_loss:.4f}, Test Acc: {test_acc:.4f}')

            if test_acc > best_accuracy:
                best_accuracy = test_acc

                torch.save(self.model.state_dict(), "output/best_moe_model.pth")

        self.model.load_state_dict(torch.load("output/best_moe_model.pth"))
        best_test_loss, best_test_acc = self.evaluate(test_loader)
        print(f'Best Test Loss: {best_test_loss:.4f}, Best Test Acc: {best_test_acc:.4f}')

        return best_test_loss, best_test_acc