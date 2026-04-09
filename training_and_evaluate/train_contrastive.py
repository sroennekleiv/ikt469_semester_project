import torch
import torch.nn as nn

class ContrastiveTrainerAndEvaluator:
    def __init__(self, model, margin=1.0, device='cuda'):
        self.model = model
        self.device = device
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=1e-3)
        
        # Contrastive loss function to encourage similar samples to be close and dissimilar samples to be apart in the embedding space
        self.criterion = nn.CosineEmbeddingLoss(margin=margin)

    def train(self, train_loader):
        self.model.train()
        total_loss = 0

        for x, y in train_loader:
            # Generate pairs of samples and their corresponding labels (1 for similar, 0 for dissimilar)
            pairs, targets = self.model.get_label_pairs(x, y)

            # Convert pairs to tensors
            pairs = [(p[0].unsqueeze(0), p[1].unsqueeze(0)) for p in pairs]
            targets = targets.to(self.device)

            self.optimizer.zero_grad()
            losses = []

            # Compute the contrastive loss for each pair of samples
            for (x1, x2), target in zip(pairs, targets):
                _, p1 = self.model(x1)
                _, p2 = self.model(x2)
                loss = self.criterion(p1, p2, target.unsqueeze(0))
                losses.append(loss)

            batch_loss = torch.stack(losses).mean()
            batch_loss.backward()
            self.optimizer.step()

            total_loss += batch_loss.item() * x.size(0)
        
        avg_loss = total_loss / len(train_loader.dataset)
        return avg_loss
    
    def evaluate(self, test_loader):
        self.model.eval()

        total_loss, correct, total = 0, 0, 0
        
        with torch.no_grad():
            for x, y in test_loader:
                x, y = x.to(self.device), y.to(self.device)

                pairs, targets = self.model.get_label_pairs(x, y)
                pairs = [(p[0].unsqueeze(0), p[1].unsqueeze(0)) for p in pairs]
                targets = targets.to(self.device)

                losses = []
                for (x1, x2), target in zip(pairs, targets):
                    _, p1 = self.model(x1)
                    _, p2 = self.model(x2)
                    loss = self.criterion(p1, p2, target.unsqueeze(0))
                    losses.append(loss)

                batch_loss = torch.stack(losses).mean()
                total_loss += batch_loss.item() * x.size(0)

                # For evaluation, we can compute accuracy based on the similarity of embeddings
                with torch.no_grad():
                    for (x1, x2), target in zip(pairs, targets):
                        _, p1 = self.model(x1)
                        _, p2 = self.model(x2)
                        similarity = torch.cosine_similarity(p1, p2)
                        predicted = (similarity > 0.5).float()  # Threshold for similarity
                        correct += (predicted == target).sum().item()
                        total += target.size(0)
            
        avg_loss = total_loss / len(test_loader.dataset)
        accuracy = correct / total
        return avg_loss, accuracy

    def run_experiment(self, train_loader, test_loader, epochs):
        for epoch in range(epochs):
            train_loss = self.train(train_loader)
            print(f'Epoch {epoch+1}/{epochs}, Train Loss: {train_loss:.4f}')
        
        test_loss, test_acc = self.evaluate(test_loader)
        print(f'Test Loss: {test_loss:.4f}, Test Accuracy: {test_acc:.4f}')
        return test_loss, test_acc