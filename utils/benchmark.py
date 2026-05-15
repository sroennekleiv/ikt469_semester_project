import os
import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F

from sklearn.metrics import silhouette_score, accuracy_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

from utils.data_preprocess import PreProcessingClass

class ExperimentBenchmarker:
    def __init__(self, device):
        super().__init__()
        self.device = device
        self.preprocessor = PreProcessingClass(size=32)

    # Function to extract embeddings from the model
    def extract_embeddings(self, model, data_loader, num_classes=10):
        model.eval()

        # List to store the embeddings and labels
        embeddings = []
        labels = []
        images = []

        with torch.no_grad():
            for x, y in data_loader:
                # Check if the data is raw or preprocessed
                if x.dtype == torch.uint8:
                    x_preprocessed = torch.stack([
                        self.preprocessor.preprocess(img.cpu(), augment=False) for img in x
                    ]).float().to(self.device)
                else:
                    x_preprocessed = x.float().to(self.device)

                y = y.to(self.device)

                print("Before reshape:", x.shape)

                if x.dim() == 2:
                    x = x.unsqueeze(0).unsqueeze(0)

                elif x.dim() == 3:
                    x = x.unsqueeze(1)

                elif x.dim() == 1:
                    x = x.view(1, 1, 28, 28)

                x = F.interpolate(x, size=(32, 32), mode='bilinear', align_corners=False)
                

                output = model(x_preprocessed) # Extract embeddings from the model

                if isinstance(output, dict):
                    z = output["embedding"]
                elif isinstance(output, tuple):
                    z = output[0]
                else:
                    z = output

                z = z.view(z.size(0), -1)
                
                embeddings.append(z.cpu().detach())
                labels.append(y.cpu().detach())
                images.append(x_preprocessed.cpu().detach())
        
        embeddings = torch.cat(embeddings, dim=0)
        labels = torch.cat(labels, dim=0)
        images = torch.cat(images, dim=0)

        embeddings = F.normalize(embeddings, dim=1)

        if num_classes < len(labels.unique()):
            selected_classes = torch.randperm(len(labels.unique()))[:num_classes]
            mask = torch.isin(labels, selected_classes)
            embeddings = embeddings[mask]
            labels = labels[mask]
            images = images[mask]

        return embeddings, labels, images
    
    def get_knn_classification(self, embeddings, labels):
        knn_results = {}

        for k in [5, 10, 20]:
            knn = KNeighborsClassifier(n_neighbors=k)
            knn.fit(embeddings, labels)
            knn_acc = knn.score(embeddings, labels)

            knn_results[f'{k}'] = knn_acc
            
        return knn_results
    
    def get_silhuoette_score(self, embeddings, labels):
        if len(embeddings) > 500:
            sample_idx = np.random.choice(len(embeddings), 500, replace=False)
            emb_sample = embeddings[sample_idx]
            labels_sample = labels[sample_idx]
        else:
            emb_sample = embeddings
            labels_sample = labels
 
        sil_score = silhouette_score(emb_sample, labels_sample)

        return sil_score
    
    def get_embedding_statistics(self, embeddings):
        mean_norm = np.linalg.norm(embeddings, axis=1).mean()
        std_norm = np.linalg.norm(embeddings, axis=1).std()
        return mean_norm, std_norm
    
    def get_per_class_clustering_statistics(self, embeddings, labels):
        intra_class_distances = []

        for c in range(10):
            class_embeddings = embeddings[labels == c]
            if len(class_embeddings) > 1:
                # Mean distance within class
                distances = []
                for i in range(len(class_embeddings)):
                    for j in range(i+1, len(class_embeddings)):
                        dist = np.linalg.norm(class_embeddings[i] - class_embeddings[j])
                        distances.append(dist)
                if distances:
                    intra_class_distances.append(np.mean(distances))
 
        if intra_class_distances:
            mean_intra = np.mean(intra_class_distances)

        return mean_intra, intra_class_distances

    def get_linear_probability(self, embeddings, labels):
        scaler = StandardScaler()
        emb_scaled = scaler.fit_transform(embeddings)

        log_reg = LogisticRegression(max_iter=1000)
        log_reg.fit(emb_scaled, labels)

        linear_acc = log_reg.score(emb_scaled, labels)
        return linear_acc
    
    def run_experiment_functions(self, embeddings, labels):
        # Compute the KNN accuracy
        knn_acc = self.get_knn_classification(embeddings=embeddings, labels=labels)

        # Compute the silhouette score
        silhouette_score = self.get_silhuoette_score(embeddings=embeddings, labels=labels)

        # Calculate the embedding statistics
        mean_norm, std_norm = self.get_embedding_statistics(embeddings=embeddings)

        # Compute per-class clustering statistics
        mean_intra, intra_class_distances = self.get_per_class_clustering_statistics(embeddings=embeddings, labels=labels)

        # Calculate the linear probability performance
        linear_acc = self.get_linear_probability(embeddings=embeddings, labels=labels)

        return {
            'knn_accuracy': knn_acc,
            'silhouette_score': silhouette_score,
            'mean_norm': mean_norm,
            'std_norm': std_norm,
            'mean_intra_distance': mean_intra,
            'linear_probe_accuracy': linear_acc,
            'intra_class_distances': intra_class_distances
        }
    
    def evaluate_test_accuracy_CLIP(self, model, data_loader):
        model.eval()

        total_correct = 0
        total_samples = 0

        with torch.no_grad():
            for x, y in data_loader:
                x = x.float().to(self.device)
                y = y.long().to(self.device)

                if x.dim() == 2:
                    x = x.unsqueeze(0).unsqueeze(0)

                elif x.dim() == 3:
                    x = x.unsqueeze(1)

                elif x.dim() == 1:
                    x = x.view(1, 1, 28, 28)

                x = F.interpolate(x, size=(32, 32), mode='bilinear', align_corners=False)

                outputs = model(x)
                logits = outputs["logits"]

                predictions = logits.argmax(dim=1)

                total_correct += predictions.eq(y).sum().item()
                total_samples += x.size(0)

        acc = total_correct / total_samples
        return acc

class VisualizeEmbeddings:
    def __init__(self):
        pass

    def visualize(self, model_name='None', embeddings=None, labels=None):
        os.makedirs("plots", exist_ok=True)

        pca = PCA(n_components=min(50, embeddings.shape[1]))
        embeddings_pca = pca.fit_transform(embeddings)

        # t-SNE visualization to reduce the dimensionality of the embeddings to 2D
        tsne = TSNE(
            n_components=2,
            perplexity=30,
            init='pca',
            random_state=42,
            learning_rate='auto'
        )

        embeddings_2d = tsne.fit_transform(embeddings_pca) # 2D dimensionality reduction

        plt.figure(figsize=(10, 10))
        scatter = plt.scatter(embeddings_2d[:, 0], embeddings_2d[:, 1], c=labels, cmap='tab10', alpha=0.7)

        plt.colorbar(scatter)
        plt.title("t-SNE Visualization of Embeddings")
        plt.xlabel("Dimension 1")
        plt.ylabel("Dimension 2")
        plt.tight_layout()
        plt.savefig(f"plots/embeddings_visualization_{model_name}.png")