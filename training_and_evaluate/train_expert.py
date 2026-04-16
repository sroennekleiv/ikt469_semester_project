import torch
import torch.nn as nn

class ExpertTrainerAndEvaluator:
    def __init__(self, model, device):
        self.model = model.to(device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=1e-3)
        self.criterion = nn.CrossEntropyLoss()
        self.device = device

    def train(self, train_loader):
        self.model.train()
        total_loss, correct, total = 0, 0, 0

        for x, y in train_loader:
            x, y = x.to(self.device), y.to(self.device)

            self.optimizer.zero_grad()
            _ , outputs = self.model(x)
            loss = self.criterion(outputs, y)
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
                _ , outputs = self.model(x)
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
            print(f'Epoch {epoch+1}/{epochs} - Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}')
        
        test_loss, test_acc = self.evaluate(test_loader)
        print(f'Test Loss: {test_loss:.4f}, Test Acc: {test_acc:.4f}')

        return test_loss, test_acc