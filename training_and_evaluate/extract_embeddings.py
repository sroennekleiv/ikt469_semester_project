import os
import torch
import torch.nn.functional as F

from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

class ExtractEmbeddings:
    def __init__(self, device):
        self.device = device

    # Function to extract embeddings from the model
    def extract_embeddings(self, model, data_loader, num_classes=10):
        model.eval()

        # List to store the embeddings and labels
        embeddings = []
        labels = []
        images = []

        with torch.no_grad():
            for x, y in data_loader:
                x = x.to(self.device)
                _, z = model(x) # Extract embeddings from the model
                embeddings.append(z.cpu())
                labels.append(y.cpu())
                images.append(x.cpu())
        
        embeddings = torch.cat(embeddings, dim=0)
        labels = torch.cat(labels, dim=0)
        images = torch.cat(images, dim=0)

        embeddings = F.normalize(embeddings, dim=1)

        # Reduce number of classes for visualization
        if num_classes < len(labels.unique()):
            selected_classes = torch.randperm(len(labels.unique()))[:num_classes]
            mask = torch.isin(labels, selected_classes)
            embeddings = embeddings[mask]
            labels = labels[mask]
            images = images[mask]

        return embeddings, labels, images
    
    # Function to compute cosine similarity and get top-k neighbors
    def get_top_k_neighbors(self, query_embedding, all_embeddings, k=5):
        cosine_similarities = F.cosine_similarity(query_embedding.unsqueeze(0), all_embeddings)
        top_k_indices = torch.topk(cosine_similarities, k=k+1).indices[1:] # Exclude the query itself
        return top_k_indices
    
    def show_neighbors(self, query_image, neighbor_images, title="Neighbors"):
        n = len(neighbor_images) + 1

        plt.figure(figsize=(3 * n, 3))
        plt.subplot(1, n, 1)
        plt.imshow(query_image.permute(1, 2, 0))
        plt.title("Query Image")
        plt.axis('off')

        for i, neighbor in enumerate(neighbor_images):
            plt.subplot(1, n, i + 2)
            plt.imshow(neighbor.permute(1, 2, 0))
            plt.title(f"Neighbor {i+1}")
            plt.axis('off')
        
        plt.suptitle(title)
        plt.tight_layout()
        plt.show()
    
    def visualize_single_embedding(self, embedding, label):
        plt.figure(figsize=(10, 10))
        plt.scatter(embedding[:, 0], embedding[:, 1], c=label, cmap='tab10', alpha=0.7)
        plt.colorbar()
        plt.title("t-SNE Visualization of Embeddings")
        plt.xlabel("Dimension 1")
        plt.ylabel("Dimension 2")
        plt.tight_layout()
        plt.show()
    
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
        plt.show()