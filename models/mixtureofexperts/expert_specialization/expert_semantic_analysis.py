import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def plot_expert_semantics(routing_weights, labels, class_names):

    num_experts = routing_weights.shape[1]
    num_classes = len(class_names)

    matrix = np.zeros((num_experts, num_classes))

    for expert in range(num_experts):

        expert_scores = routing_weights[:, expert]

        top_indices = np.argsort(expert_scores)[::-1][:500]

        top_labels = labels[top_indices]

        for cls in range(num_classes):
            matrix[expert, cls] = np.mean(top_labels == cls)

    plt.figure(figsize=(12, 4))

    sns.heatmap(
        matrix,
        annot=True,
        cmap="Blues",
        xticklabels=class_names,
        yticklabels=[f"Expert {i}" for i in range(num_experts)]
    )

    plt.xlabel("Class", fontsize=16)
    plt.ylabel("Expert", fontsize=16)

    plt.title(
        "Semantic Specialization of Experts",
        fontsize=20,
        pad=20
    )

    plt.tight_layout()
    plt.show()