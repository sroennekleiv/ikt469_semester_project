import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from sklearn.metrics.pairwise import cosine_similarity

from utils.dataset import FashionMNISTDataset

from models.mixtureofexperts.mixture_of_experts import ( MixtureOfExperts )

from training_and_evaluate.train_expert import ( ExpertTrainerAndEvaluator )


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Load the dataset and extract class names
dataset = FashionMNISTDataset(batch_size=64, size=32, split_fraction=0.5)
train_loader, test_loader = dataset.get_dataloaders(is_contrastive=False)
class_names = dataset.get_class_names()

# Load the MixtureOfExpert model
moe_model = MixtureOfExperts(num_classes=len(class_names), embedding_dim=128, projection_dim=64)
trainer = ExpertTrainerAndEvaluator(moe_model, device)

num_epochs = 20

for epoch in range(num_epochs):
    train_loss, train_acc = trainer.train(train_loader)
    trainer.scheduler.step()
    test_loss, test_acc = trainer.evaluate(test_loader)
    if (epoch + 1) % 3 == 0:
        print(f'Epoch {epoch+1}/{num_epochs} - Loss: {train_loss:.4f}, Test Acc: {test_acc:.4f}')

routing_weights_all = []
labels_all = []

fused_embeddings_all = []

expert_0_embeddings = []
expert_1_embeddings = []
expert_2_embeddings = []

predictions_all = []

moe_model.eval()

with torch.no_grad():

    for x, y in test_loader:

        x = x.to(device)
        y = y.to(device)

        outputs = moe_model(
            x,
            return_routing=True
        )

        logits = outputs["logits"]

        routing_weights = outputs["routing_weights"]

        fused_embeddings = outputs["embedding"]

        expert_embeddings = outputs["expert_embeddings"]

        predictions = logits.argmax(dim=1)

        routing_weights_all.append(
            routing_weights.cpu()
        )

        fused_embeddings_all.append(
            fused_embeddings.cpu()
        )

        expert_0_embeddings.append(
            expert_embeddings[:, 0].cpu()
        )

        expert_1_embeddings.append(
            expert_embeddings[:, 1].cpu()
        )

        expert_2_embeddings.append(
            expert_embeddings[:, 2].cpu()
        )

        labels_all.append(y.cpu())

        predictions_all.append(predictions.cpu())


routing_weights_all=torch.cat(routing_weights_all).numpy()
labels_all=torch.cat(labels_all).numpy()
predictions_all=torch.cat(predictions_all).numpy()

fused_embeddings_all=torch.cat(
    fused_embeddings_all
).numpy()

expert_0_embeddings=torch.cat(
    expert_0_embeddings
).numpy()

expert_1_embeddings=torch.cat(
    expert_1_embeddings
).numpy()

expert_2_embeddings=torch.cat(
    expert_2_embeddings
).numpy()

num_experts=3
epsilon=1e-10

# ============================================================
# ROUTING STATS
# ============================================================

print("\nRouting Weight Stats")
print("---------------------")
print("Min:", routing_weights_all.min())
print("Max:", routing_weights_all.max())
print("Mean:", routing_weights_all.mean())

# ============================================================
# ENTROPY
# ============================================================

entropy_per_sample=-np.sum(
    routing_weights_all*
    np.log(routing_weights_all+epsilon),
    axis=1
)

mean_entropy=entropy_per_sample.mean()

max_entropy=np.log(num_experts)

specialization_score=np.clip(
    1-(mean_entropy/max_entropy),
    0,
    1
)

print(f"\nMean entropy (learned): {mean_entropy:.4f}")
print(f"Max entropy (random): {max_entropy:.4f}")

print(
    f"\nSPECIALIZATION SCORE: "
    f"{specialization_score:.1%}"
)

if specialization_score>0.70:
    print("Strong expert specialization")

elif specialization_score>0.40:
    print("Moderate expert specialization")

else:
    print("Weak specialization / near random routing")

# ============================================================
# SOFT EXPERT USAGE
# ============================================================

expert_usage=routing_weights_all.mean(axis=0)

print("\nSoft Expert Usage")

for i,usage in enumerate(expert_usage):

    print(
        f"Expert {i}: "
        f"{usage:.2%}"
    )

# ============================================================
# HARD EXPERT ASSIGNMENT
# ============================================================

hard_assignments=np.argmax(
    routing_weights_all,
    axis=1
)

hard_usage=np.bincount(
    hard_assignments,
    minlength=num_experts
)/len(hard_assignments)

print("\nHard Expert Assignment")

for i,usage in enumerate(hard_usage):

    print(
        f"Expert {i}: "
        f"{usage:.2%}"
    )

collapse_score=hard_usage.max()

print(
    f"\nCollapse Score: "
    f"{collapse_score:.2%}"
)

if collapse_score>0.80:
    print("Severe expert collapse")

elif collapse_score>0.60:
    print("Moderate collapse")

else:
    print("Healthy expert utilization")

balance_std=np.std(expert_usage)

print(f"\nUsage Std: {balance_std:.4f}")

# ============================================================
# CLASS → EXPERT ANALYSIS
# ============================================================

class_expert_matrix=np.zeros(
    (len(class_names),num_experts)
)

for class_id in range(len(class_names)):

    class_mask=labels_all==class_id

    class_routing=routing_weights_all[
        class_mask
    ]

    class_expert_matrix[class_id]=(
        class_routing.mean(axis=0)
    )

print("\nDominant Expert Per Class")

for class_id,class_name in enumerate(class_names):

    class_mask=labels_all==class_id

    dominant=np.argmax(
        routing_weights_all[class_mask],
        axis=1
    )

    counts=np.bincount(
        dominant,
        minlength=num_experts
    )

    dominant_expert=np.argmax(counts)

    percentage=(
        counts[dominant_expert]
        /counts.sum()
    )

    print(
        f"{class_name:15s} "
        f"→ Expert {dominant_expert} "
        f"({percentage:.2%})"
    )

# ============================================================
# ROUTING CONFIDENCE
# ============================================================

routing_confidence=routing_weights_all.max(axis=1)

mean_confidence=routing_confidence.mean()

print(
    f"\nMean Routing Confidence: "
    f"{mean_confidence:.2%}"
)

if mean_confidence>0.75:
    print("Router is highly confident")

elif mean_confidence>0.50:
    print("Router shows moderate confidence")

else:
    print("Router is uncertain")

# ============================================================
# EMBEDDING SPECIALIZATION
# ============================================================

sim_01=np.mean(
    np.sum(
        expert_0_embeddings*
        expert_1_embeddings,
        axis=1
    )
)

sim_02=np.mean(
    np.sum(
        expert_0_embeddings*
        expert_2_embeddings,
        axis=1
    )
)

sim_12=np.mean(
    np.sum(
        expert_1_embeddings*
        expert_2_embeddings,
        axis=1
    )
)

print(f"\nExpert0 ↔ Expert1: {sim_01:.4f}")
print(f"Expert0 ↔ Expert2: {sim_02:.4f}")
print(f"Expert1 ↔ Expert2: {sim_12:.4f}")

average_similarity=(
    sim_01+sim_02+sim_12
)/3

print(
    f"\nAverage Similarity: "
    f"{average_similarity:.4f}"
)

if average_similarity<0.30:
    print("Strong feature specialization")

elif average_similarity<0.60:
    print("Moderate specialization")

else:
    print("Experts learn very similar features")

# ============================================================
# ROUTING CONSISTENCY
# ============================================================

print("\nClass Routing Consistency")

for class_id,class_name in enumerate(class_names):

    class_mask=labels_all==class_id

    class_routes=routing_weights_all[
        class_mask
    ]

    dominant_experts=class_routes.argmax(axis=1)

    consistency=(
        np.bincount(
            dominant_experts,
            minlength=num_experts
        ).max()
        /
        len(dominant_experts)
    )

    print(
        f"{class_name:15s} "
        f"| Consistency: {consistency:.2%}"
    )

# ============================================================
# PCA
# ============================================================

pca=PCA(n_components=2)

reduced_embeddings=pca.fit_transform(
    fused_embeddings_all
)

plt.figure(figsize=(10,8))

scatter=plt.scatter(
    reduced_embeddings[:,0],
    reduced_embeddings[:,1],
    c=labels_all,
    cmap="tab10",
    alpha=0.7
)

plt.legend(
    handles=scatter.legend_elements()[0],
    labels=class_names,
    bbox_to_anchor=(1.05,1),
    loc="upper left"
)

plt.title("PCA of Fused Embeddings")

plt.tight_layout()

plt.savefig(
    "pca_fused_embeddings.png",
    dpi=300
)

plt.close()

# ============================================================
# PCA BY EXPERT
# ============================================================

plt.figure(figsize=(10,8))

scatter=plt.scatter(
    reduced_embeddings[:,0],
    reduced_embeddings[:,1],
    c=hard_assignments,
    cmap="Set1",
    alpha=0.7
)

plt.title(
    "Latent Embeddings Colored by Expert"
)

plt.tight_layout()

plt.savefig(
    "expert_specialization_pca.png",
    dpi=300
)

plt.close()

# ============================================================
# TSNE
# ============================================================

subset_size=min(
    2000,
    len(fused_embeddings_all)
)

indices=np.random.choice(
    len(fused_embeddings_all),
    subset_size,
    replace=False
)

subset_embeddings=fused_embeddings_all[
    indices
]

subset_labels=labels_all[
    indices
]

tsne=TSNE(
    n_components=2,
    perplexity=30,
    random_state=42
)

tsne_embeddings=tsne.fit_transform(
    subset_embeddings
)

plt.figure(figsize=(10,8))

scatter=plt.scatter(
    tsne_embeddings[:,0],
    tsne_embeddings[:,1],
    c=subset_labels,
    cmap="tab10",
    alpha=0.7
)

plt.legend(
    handles=scatter.legend_elements()[0],
    labels=class_names,
    bbox_to_anchor=(1.05,1),
    loc="upper left"
)

plt.title("t-SNE of Learned Embeddings")

plt.tight_layout()

plt.savefig(
    "tsne_embeddings.png",
    dpi=300
)

plt.close()