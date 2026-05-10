import torch
import torch.nn as nn
import torch.nn.functional as F

from torchvision import transforms

from utils.data_preprocess import PreProcessingClass

class SimCLRLoss(nn.Module):
    def __init__(self, temperature=0.07):
        super().__init__()
        self.temperature = temperature # Temperature parameter to control the scale of the similarity scores in the contrastive loss

    def forward(self, z1, z2):

        batch_size = z1.size(0)

        z1 = F.normalize(z1, dim=-1)
        z2 = F.normalize(z2, dim=-1)

        representations = torch.cat([z1, z2], dim=0)

        similarity_matrix = torch.matmul(representations, representations.T)
        mask = torch.eye(2 * batch_size, device=z1.device).bool()
        similarity_matrix = similarity_matrix.masked_fill(mask, -1e9)

        positives = torch.cat([
            torch.diag(similarity_matrix, batch_size),
            torch.diag(similarity_matrix, -batch_size)
        ], dim=0)

        numerator = torch.exp(positives / self.temperature)

        denominator = torch.exp(similarity_matrix / self.temperature).sum(dim=1)

        loss = -torch.log(numerator / denominator)

        return loss.mean()

class ContrastiveTrainerAndEvaluator:
    def __init__(self, model, temperature=0.5, device='cuda'):
        self.model = model
        self.device = device
        
        self.criterion = SimCLRLoss(temperature=temperature) # Loss function to compute the contrastive loss between the representations of the original and augmented images
        self.classification_criterion = nn.CrossEntropyLoss() # Loss function to compute the classification loss for the supervised component of the training
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=1e-3, weight_decay=1e-4)
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=50)

        self.scaler = torch.amp.GradScaler(enabled=(device.type=="cuda"))

        self.augmentation = transforms.Compose([
            transforms.RandomResizedCrop(32, scale=(0.7, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomApply([transforms.RandomRotation(15)], p=0.5),
            transforms.RandomApply([transforms.GaussianBlur(kernel_size=3)], p=0.3)
        ])

    def perform_augmentation(self, x):
        augmented = []
        for image in x:
            image = image.cpu()

            # Ensure the image has 3 channels by repeating the single channel if it's grayscale
            if image.dim() == 2:
                image = image.unsqueeze(0)

            aug_img = self.augmentation(image.cpu())
            augmented.append(aug_img)
        
        augmented = torch.stack(augmented, dim=0).to(self.device)
        return augmented

    def train(self, train_loader):
        self.model.train()
        
        total_loss = 0
        total_samples = 0
        skip_batch = 0

        total_correct = 0

        for x, y in train_loader:
            x = x.float()
            y = y.long().to(self.device)

            # Generate two augmented views of the input images for contrastive learning
            view1 = self.perform_augmentation(x)
            view2 = self.perform_augmentation(x)

            self.optimizer.zero_grad()

            with torch.amp.autocast('cuda', enabled=(self.device.type=="cuda")):
                # Extract the output from the model for both the original and augmented images
                outs1 = self.model(view1)
                outs2 = self.model(view2)

                # Extract the projection and classification outputs from the model's output dictionaries for both views
                proj1 = outs1["projection"]
                proj2 = outs2["projection"]

                logits = outs1["logits"]

                contrastive_loss = self.criterion(proj1, proj2)
                class_loss = self.classification_criterion(logits, y)

                loss = contrastive_loss + 0.1 * class_loss # Combine the contrastive loss and classification loss with a weighting factor to balance their contributions to the overall loss

            if not torch.isfinite(loss) or loss.item() == 0.0:
                skip_batch += 1
                continue

            # Perform backpropagation and optimization steps with gradient scaling
            self.scaler.scale(loss).backward()
            self.scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)

            # Update the model parameters based on the computed gradients
            self.scaler.step(self.optimizer)
            self.scaler.update()

            predictions = logits.argmax(dim=1)

            total_correct += predictions.eq(y).sum().item()
            total_loss += loss.item() * x.size(0)
            total_samples += x.size(0)

        avg_loss = total_loss / total_samples
        avg_acc = total_correct / total_samples
        return avg_loss, avg_acc
    
    def evaluate(self, test_loader):
        self.model.eval()
        
        total_loss = 0
        total_correct = 0
        total_samples = 0

        with torch.no_grad():
            for x, y in test_loader:
                x = x.float()
                y = y.long().to(self.device)

                print("Before reshape:", x.shape)

                if x.dim() == 2:
                    x = x.unsqueeze(0).unsqueeze(0)

                elif x.dim() == 3:
                    x = x.unsqueeze(1)

                elif x.dim() == 1:
                    x = x.view(1, 1, 28, 28)

                x = F.interpolate(x, size=(32, 32), mode='bilinear', align_corners=False)

                outputs = self.model(x)
                logits = outputs["logits"]
                loss = self.classification_criterion(logits, y)

                predictions = logits.argmax(dim=1)

                total_loss += loss.item() * x.size(0)
                total_correct += predictions.eq(y).sum().item()
                total_samples += x.size(0)

        avg_loss = total_loss / total_samples
        acc = total_correct / total_samples
        
        return avg_loss, acc

    def run_experiment(self, train_loader, test_loader, epochs):
        for epoch in range(epochs):
            train_loss, acc = self.train(train_loader)
            print(f'Epoch {epoch+1}/{epochs}, Train Loss: {train_loss:.4f}, Acc: {acc * 100:.2f}')

            self.scheduler.step()
        
        test_loss, acc = self.evaluate(test_loader)
        print(f'Test Loss: {test_loss:.4f}, Acc: {acc:.4f}')
        return test_loss