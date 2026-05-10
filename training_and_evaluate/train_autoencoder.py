import torch
import torch.nn as nn
import torch.nn.functional as F

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class AutoEncoderTrainerAndEvaluator:
    def __init__(self, model, lr=1e-3):
        self.model = model.to(device)

        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr, weight_decay=1e-4)
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=50)
        self.criterion = nn.MSELoss()

        self.device = device

    def train(self, train_loader):
        self.model.train()
        
        total_loss = 0.0

        total_samples = 0

        for x, _ in train_loader:
            x = x.to(self.device)

            self.optimizer.zero_grad()

            # Extraxt the output from the model for both the original and augmented images
            out1 = self.model(x)
            reconstruction = out1["reconstruction"]

            # Compute the reconstruction loss between the input and the reconstructed output
            loss = self.criterion(reconstruction, x)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.optimizer.step()

            total_loss += loss.item() * x.size(0)
            total_samples += x.size(0)

        avg_loss = total_loss / total_samples
        return avg_loss

    def evaluate(self, test_loader):
        self.model.eval()

        total_loss = 0.0
        total_samples = 0

        with torch.no_grad():
            for x, _ in test_loader:
                x = x.to(self.device)
                outs = self.model(x)
                
                reconstruction = outs["reconstruction"]
                loss = self.criterion(reconstruction, x)

                total_loss += loss.item() * x.size(0)
                total_samples += x.size(0)

        avg_loss = total_loss / total_samples

        return avg_loss
    
    def run_experiment(self, train_loader, test_loader, epochs=10):
        best_loss = 0.0

        for epoch in range(epochs):
            train_loss = self.train(train_loader)
            self.scheduler.step()
            print(f"Epoch {epoch+1}/{epochs} - Train Loss: {train_loss:.4f}")

            if train_loss < best_loss:
                torch.save(self.model.state_dict(), "best_autoencoder.pth")
        
        test_loss = self.evaluate(test_loader)
        print(f"Test Loss: {test_loss:.4f}")
        return test_loss
