import torch
import torch.nn as nn
import torch.nn.functional as F

from training_and_evaluate.extract_embeddings import ExtractEmbeddings
from utils.data_preprocess import PreProcessingClass

class SupervisedContrastiveLoss(nn.Module):
    def __init__(self, temperature=0.07):
        super().__init__()
        self.temperature = temperature # Controls seperation of positive / negative pairs
        self.extractor = ExtractEmbeddings(device='cuda' if torch.cuda.is_available() else 'cpu')

    def forward(self, features, labels):
        device = features.device
        batch_size = features.shape[0]
        n_views = features.shape[1]

        features = F.normalize(features, dim=2)
        features = features.view(batch_size * n_views, -1) # (batch_size * n_views, embedding_dim)

        # Create mask for positive pairs (same class) and negative pairs (different classes)
        labels = labels.contiguous().view(-1, 1)
        mask = torch.eq(labels, labels.T).float().to(device)

        logits = torch.matmul(features, features.T) / self.temperature # (batch_size * n_views, batch_size * n_views)
        logits_max, _ = torch.max(logits, dim=1, keepdim=True)
        logits = logits - logits_max.detach()

        mask = mask.repeat(n_views, n_views) # (batch_size * n_views, batch_size * n_views)

        logits_masked = torch.ones_like(mask)
        logits_masked.fill_diagonal_(0) # Exclude self-similarity
        mask = mask * logits_masked # Mask out self-similarity

        exp_logits = torch.exp(logits) * logits_masked
        log_prob = logits - torch.log(exp_logits.sum(dim=1, keepdim=True))

        mask_sum = mask.sum(dim=1)
        mask_sum[mask_sum == 0] = 1.0 # Avoid division by zero for samples with no positive pairs

        # Compute mean loss for positive pairs
        mean_log_prob_pos = (mask * log_prob).sum(dim=1) / mask_sum

        # Return the mean loss over the batch
        loss = -mean_log_prob_pos.mean()
        return loss

class ContrastiveTrainerAndEvaluator:
    def __init__(self, model, margin=0.07, device='cuda'):
        self.model = model
        self.device = device
        
        self.criterion = SupervisedContrastiveLoss(temperature=margin)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=1e-3)

        self.preprocessor = PreProcessingClass(size=28)

    def get_two_views(self, x):
        view1 = torch.stack([self.preprocessor.preprocess(img.cpu(), augment=True) for img in x]).to(self.device)
        view2 = torch.stack([self.preprocessor.preprocess(img.cpu(), augment=True) for img in x]).to(self.device)
        return view1, view2

    def train(self, train_loader):
        self.model.train()
        total_loss = 0.0
        total_batches = 0

        for x, y in train_loader:
            x, y = x.to(self.device), y.to(self.device)

            # Generate pairs of samples and their corresponding labels (1 for similar, -1 for dissimilar)
            view1, view2 = self.get_two_views(x)

            self.optimizer.zero_grad()

            z1, p1 = self.model(view1)
            z2, p2 = self.model(view2)

            features = torch.stack([p1, p2], dim=1)
            loss = self.criterion(features, y)
            loss.backward()
            self.optimizer.step()

            total_loss += loss.item() * x.size(0)
            total_batches += 1

        avg_loss = total_loss / max(total_batches, 1)
        return avg_loss
    
    def evaluate(self, test_loader):
        self.model.eval()
        total_loss = 0.0
        correct, total = 0, 0
        total_batches = 0

        with torch.no_grad():
            for x, y in test_loader:
                x, y = x.to(self.device), y.to(self.device)

                view1, view2 = self.get_two_views(x)

                z1, p1 = self.model(view1)
                z2, p2 = self.model(view2)

                features = torch.stack([p1, p2], dim=1)
                loss = self.criterion(features, y)

                correct += (p1.argmax(dim=1) == y).sum().item() # Using projection head for evaluation
                total += y.size(0)

                total_loss += loss.item() * x.size(0)
                total_batches += 1

        avg_loss = total_loss / max(total_batches, 1)
        accuracy = correct / max(total, 1) * 100.0
        return avg_loss, accuracy

    def run_experiment(self, train_loader, test_loader, epochs):
        for epoch in range(epochs):
            train_loss = self.train(train_loader)
            print(f'Epoch {epoch+1}/{epochs}, Train Loss: {train_loss:.4f}')
        
        test_loss, test_acc = self.evaluate(test_loader)
        print(f'Test Loss: {test_loss:.4f}, Test Accuracy: {test_acc:.4f}')
        return test_loss, test_acc