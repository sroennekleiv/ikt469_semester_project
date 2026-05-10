import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from models.mixtureofexperts.expert_specialization.expert_specialization_loss_functions import orthogonality_loss, routing_sharpness_loss, expert_consistency_loss, routing_entropy_loss, loading_balancing_loss

class ExpertTrainerAndEvaluator:
    def __init__(self, model, device, lr=1e-3, contrastive_weight=0.1, entropy_weight=0.01, load_balance_weight=0.01):
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

    def nt_xent_loss(self, z1, z2, temperature=0.5):
        batch_size = z1.size(0)

        z1 = F.normalize(z1, dim=-1)
        z2 = F.normalize(z2, dim=-1)

        # Combine the sets of representations from both views
        representations = torch.cat([z1, z2], dim=0)

        # Compute the cosine similarity matrix between all pairs of representations
        similarity_matrix = torch.matmul(representations, representations.T)
        mask = torch.eye(2 * batch_size, device=self.device).bool()
        similarity_matrix = similarity_matrix.masked_fill(mask, -1e9)

        # Extract the positive pairs (where the representations come from the same image)
        positives = torch.cat([
            torch.diag(similarity_matrix, batch_size),
            torch.diag(similarity_matrix, -batch_size)
        ], dim=0)

        numerator = torch.exp(positives / temperature)
        denominator = torch.exp(similarity_matrix / temperature).sum(dim=1)
        loss = -torch.log(numerator / denominator)

        return loss.mean()


    def train(self, train_loader):
        self.model.train()

        total_loss = 0
        total_correct = 0
        total_samples = 0

        total_ce_loss = 0
        total_contrastive_loss = 0
        total_entropy_loss = 0

        for x, y in train_loader:
            x = x.to(self.device)
            y = y.to(self.device)

            self.optimizer.zero_grad()

            # Apply random noise augmentation to the input images to create augmented views
            noise = torch.randn_like(x) * 0.05
            x_augmented = torch.clamp(x + noise, 0, 1)

            outputs_1 = self.model(x, return_routing=True)
            outputs_2 = self.model(x_augmented,return_routing=True)

            # Extract the logits
            logits = outputs_1["logits"]

            # Extraxt the projection
            projection_1 = outputs_1["projection"]
            projection_2 = outputs_2["projection"]

            # Extract the routing weights and expert embeddings for computing the specialization losses
            routing_weights = outputs_1["routing_weights"]
            expert_embeddings = outputs_1["expert_embeddings"]

            specialization_score = self.compute_specialization_score(routing_weights=routing_weights)
            self.specialization_history.append(specialization_score)

            # Compute the various loss components for expert specialization and routing effectiveness
            orth_loss = orthogonality_loss(expert_embeddings)
            sharpness_loss = routing_sharpness_loss(routing_weights)
            #consistency_loss = expert_consistency_loss(expert_embeddings, routing_weights)
            entropy_loss = routing_entropy_loss(routing_weights)
            balance_loss = loading_balancing_loss(routing_weights)

            ce_loss = self.criterion(logits, y)

            contrastive_loss = self.nt_xent_loss(projection_1, projection_2)

            # Combine all the loss components with their respective weighting factors to compute the total loss for backpropagation
            loss = (
                ce_loss
                + self.contrastive_weight * contrastive_loss
                + self.orthogonality_weight * orth_loss
                #+ self.sharpness_weight * sharpness_loss
                #+ self.consistency_weight * consistency_loss
                #+ self.entropy_weight * entropy_loss
                + self.load_balance_weight * balance_loss
            )

            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()
            total_loss += loss.item() * x.size(0)

            total_ce_loss += ce_loss.item()
            total_contrastive_loss += contrastive_loss.item()
            total_entropy_loss += entropy_loss.item()

            predictions = logits.argmax(dim=1)
            total_correct += (predictions.eq(y).sum().item())

            total_samples += y.size(0)

            # Get the expert usage by averaging the routing weights across the batch
            expert_usage = routing_weights.mean(dim=0)
            self.routing_history.append(expert_usage.detach().cpu().numpy())

        average_loss = total_loss / total_samples
        accuracy = total_correct / total_samples

        print("\nTraining Statistics")
        print(f"CE Loss: {total_ce_loss / len(train_loader):.4f}")
        print(f"Contrastive Loss: {total_contrastive_loss / len(train_loader):.4f}")
        print(f"Entropy Loss: {total_entropy_loss / len(train_loader):.4f}")

        return average_loss, accuracy
    
    def evaluate(self, test_loader):
        self.model.eval()

        total_loss = 0
        total_correct = 0
        total_samples = 0

        expert_usage_accumulator = None

        with torch.no_grad():
            for x, y in test_loader:
                x = x.to(self.device)
                y = y.to(self.device)

                outputs = self.model(x, return_routing=True)

                logits = outputs["logits"]
                routing_weights = outputs["routing_weights"]
                loss = self.criterion(logits, y)

                total_loss += loss.item() * x.size(0)
                predictions = logits.argmax(dim=1)
                total_correct += (predictions.eq(y).sum().item())
                total_samples += y.size(0)

                expert_usage = routing_weights.mean(dim=0)

                if expert_usage_accumulator is None:
                    expert_usage_accumulator = expert_usage.detach()
                else:
                    expert_usage_accumulator += expert_usage.detach()

        average_expert_usage = (
            expert_usage_accumulator / len(test_loader)
        )

        print("\nEvaluation Expert Usage")
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
            print(f'Epoch {epoch+1}/{epochs} - Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}')

            if train_acc > best_accuracy:
                best_accuracy = train_acc

                torch.save(
                    self.model.state_dict(),
                    "output/best_moe_model.pth"
                )

            test_loss, test_acc = self.evaluate(test_loader)
            print(f'Test Loss: {test_loss:.4f}, Test Acc: {test_acc:.4f}')

        return test_loss, test_acc