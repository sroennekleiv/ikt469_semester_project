import torch
import torch.nn as nn

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class AutoEncoderTrainerAndEvaluator:
    def __init__(self, model):
        self.model = model.to(device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=1e-3)
        self.criterion = nn.MSELoss()
        self.device = device

    def train(self, train_loader):
        self.model.train()
        total_loss, total = 0.0, 0.0

        for x, _ in train_loader:
            x = x.to(self.device)

            self.optimizer.zero_grad()
            x_hat, _ = self.model(x)
            loss = self.criterion(x_hat, x)
            loss.backward()
            self.optimizer.step()

            total_loss += loss.item() * x.size(0)
            total += x.size(0)
        avg_loss = total_loss / total
        return avg_loss

    def evaluate(self, test_loader):
        self.model.eval()

        total_loss, total = 0.0, 0.0

        with torch.no_grad():
            for x, _ in test_loader:
                x = x.to(self.device)
                x_hat, _ = self.model(x)
                loss = self.criterion(x_hat, x)

                total_loss += loss.item() * x.size(0)
        avg_loss = total_loss / len(test_loader.dataset)

        return avg_loss
    
    def run_experiment(self, train_loader, test_loader, epochs=10):
        for epoch in range(epochs):
            train_loss = self.train(train_loader)
            print(f"Epoch {epoch+1}/{epochs} - Train Loss: {train_loss:.4f}")
        
        test_loss = self.evaluate(test_loader)
        print(f"Test Loss: {test_loss:.4f}")
        return test_loss
