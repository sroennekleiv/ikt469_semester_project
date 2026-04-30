import torch
import torch.nn as nn
import torch.nn.functional as F

from utils.data_preprocess import PreProcessingClass

class SimCLRLoss(nn.Module):
    def __init__(self, temperature=0.07):
        super().__init__()
        self.temperature = temperature # Controls seperation of positive / negative pairs

    def forward(self, z1, z2):
        batch_size = z1.shape[0]

        z1 = F.normalize(z1, dim=1)
        z2 = F.normalize(z2, dim=1)

        representations = torch.cat([z1, z2], dim=0)
        similarity_matrix = torch.matmul(representations, representations.T)

        labels = torch.arange(batch_size, device=z1.device)
        labels = torch.cat([labels + batch_size, labels], dim=0)

        mask = torch.eye(2 * batch_size, dtype=torch.bool, device=z1.device)

        similarity_matrix = similarity_matrix / self.temperature
        similarity_matrix = similarity_matrix.masked_fill(mask, -9e9)

        loss = F.cross_entropy(similarity_matrix, labels)
        return loss

class ContrastiveTrainerAndEvaluator:
    def __init__(self, model, temperature=0.07, device='cuda'):
        self.model = model
        self.device = device
        
        self.criterion = SimCLRLoss(temperature=temperature)
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=1e-3, weight_decay=1e-4)
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=10)

        self.preprocessor = PreProcessingClass(size=32)
        self.use_amp = self.device == "cuda"
        self.scaler = torch.amp.GradScaler(enabled=self.use_amp)


    def get_two_views(self, x, augment=False):
        view1 = torch.stack([self.preprocessor.preprocess(img.cpu(), augment=augment) for img in x]).float().to(self.device)
        view2 = torch.stack([self.preprocessor.preprocess(img.cpu(), augment=augment) for img in x]).float().to(self.device)
        return view1, view2

    def train(self, train_loader):
        self.model.train()
        total_loss = 0.0
        total_samples = 0
        skip_batch = 0

        for x, y in train_loader:
            x = x.float()
            y = y.long().to(self.device)

            # Generate two augmented views
            view1, view2 = self.get_two_views(x, augment=True)

            self.optimizer.zero_grad()

            with torch.amp.autocast(device_type='cuda', enabled=self.use_amp):
                z1, p1 = self.model(view1)
                z2, p2 = self.model(view2)

                loss = self.criterion(p1, p2)

            if not torch.isfinite(loss) or loss.item() == 0.0:
                skip_batch += 1
                continue

            self.scaler.scale(loss).backward()
            self.scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.scaler.step(self.optimizer)
            self.scaler.update()

            total_loss += loss.item() * x.size(0)
            total_samples += x.size(0)

        avg_loss = total_loss / max(total_samples, 1)
        return avg_loss
    
    def evaluate(self, test_loader):
        self.model.eval()
        total_loss = 0.0
        total_samples = 0

        with torch.no_grad():
            for x, y in test_loader:
                x = x.float()
                y = y.long().to(self.device)

                # Generate two augmented views (but without augmentation)
                view1, view2 = self.get_two_views(x, augment=False)

                z1, p1 = self.model(view1)
                z2, p2 = self.model(view2)

                loss = self.criterion(p1, p2)

                total_loss += loss.item() * x.size(0)
                total_samples += x.size(0)

        avg_loss = total_loss / max(total_samples, 1)
        
        return avg_loss

    def run_experiment(self, train_loader, test_loader, epochs):
        for epoch in range(epochs):
            train_loss = self.train(train_loader)
            print(f'Epoch {epoch+1}/{epochs}, Train Loss: {train_loss:.4f}')

            self.scheduler.step()
        
        test_loss = self.evaluate(test_loader)
        print(f'Test Loss: {test_loss:.4f}')
        return test_loss