import numpy as np
import matplotlib.pyplot as plt
import torch

# Functions to analyze expert specialization and routing patterns
def analyze_specialization(routing_weights, labels, class_names):
    num_experts = routing_weights.shape[1]

    for expert in range(num_experts):
        expert_scores = routing_weights[:, expert]

        topindices = np.argsort(expert_scores)[::-1]
        toplabels = labels[topindices[:200]]

        unique, counts = np.unique(toplabels, return_counts=True)

        print(f"\nExpert {expert}")

        sorted_counts = sorted(
            zip(unique, counts),
            key=lambda x: x[1],
            reverse=True
        )

        for id, count in sorted_counts[:5]:
            percentage = count / len(toplabels) * 100
            print(f"{class_names[id]:15s} {percentage:.2f}%")

# Plotting the evolution of routing weights for each expert over training
def plot_routing_evolution(trainer):
    history = torch.stack(trainer.routing_history).numpy()

    plt.figure(figsize=(10, 6))

    for expert_id in range(history.shape[1]):
        plt.plot(
            history[:, expert_id],
            label=f"Expert {expert_id}"
        )

    plt.xlabel("Training Step")
    plt.ylabel("Average Routing Weight")
    plt.title("Expert Routing Evolution")
    plt.legend()
    plt.tight_layout()
    plt.savefig("routing_evolution.png",dpi=300)
    plt.close()

# Plotting the specialization score over time
def plot_specialization_score(trainer):
    plt.figure(figsize=(8, 5))
    plt.plot(trainer.specialization_history, marker='o')
    plt.xlabel("Training Step")
    plt.ylabel("Specialization Score")
    plt.title("Emergent Expert Specialization")
    plt.tight_layout()
    plt.savefig("specialization_score.png",dpi=300)
    plt.close()