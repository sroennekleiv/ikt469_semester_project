"""
Gating Effectiveness Analysis with Training
============================================

This script:
1. TRAINS the MoE model with optimized load_balance_weight=0.01
2. ANALYZES gating effectiveness with 5 key metrics
3. GENERATES comprehensive visualizations

Fixed version that properly trains the model before analysis.
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import entropy as scipy_entropy

from utils.dataset import FashionMNISTDataset
from models.mixtureofexperts.mixture_of_experts import MixtureOfExperts
from training_and_evaluate.train_expert import ExpertTrainerAndEvaluator

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

print("=" * 80)
print("GATING EFFECTIVENESS ANALYSIS (with training)")
print("=" * 80)

# ============================================================================
# STEP 1: LOAD DATA
# ============================================================================

print("\nLoading data...")
dataset = FashionMNISTDataset(batch_size=64, size=32, split_fraction=0.5)
train_loader, test_loader = dataset.get_dataloaders(is_contrastive=False)
class_names = dataset.get_class_names()

# ============================================================================
# STEP 2: CREATE AND TRAIN MODEL
# ============================================================================

print("Creating and training MoE model with load_balance_weight=0.01...")
print("-" * 80)

moe_model = MixtureOfExperts(num_classes=len(class_names))
trainer = ExpertTrainerAndEvaluator(moe_model, device)

print(f"Training configuration:")
print(f"  Model: MixtureOfExperts (3 experts)")
print(f"  Load balance weight: {trainer.load_balance_weight}")
print(f"  Device: {device}")

# Train for 15 epochs
num_epochs = 15
for epoch in range(num_epochs):
    train_loss, train_acc = trainer.train(train_loader)
    trainer.scheduler.step()
    test_loss, test_acc = trainer.evaluate(test_loader)
    if (epoch + 1) % 3 == 0:
        print(f'Epoch {epoch+1:2d}/{num_epochs} - Loss: {train_loss:.4f}, Test Acc: {test_acc:.4f}')

print("-" * 80)
print(f"✓ Training complete! Final test accuracy: {test_acc:.4f}\n")

# ============================================================================
# STEP 3: COLLECT ROUTING WEIGHTS
# ============================================================================

print("=" * 80)
print("GATING EFFECTIVENESS ANALYSIS")
print("=" * 80)

routing_weights_list = []
all_labels = []

with torch.no_grad():
    for batch_images, batch_labels in test_loader:
        batch_images = batch_images.to(device).float()
        x = moe_model.stem(batch_images)
        routing_weights = moe_model.experts.router(x)

        routing_weights_list.append(routing_weights.cpu())
        all_labels.append(batch_labels.cpu())

routing_weights_all = torch.cat(routing_weights_list).numpy()
all_labels = torch.cat(all_labels).numpy()

num_experts = 3

# ============================================================================
# METRIC 1: GATING ENTROPY SCORE
# ============================================================================

print("\n" + "=" * 80)
print("METRIC 1: GATING ENTROPY SCORE")
print("=" * 80)

epsilon = 1e-10
entropy_per_sample = -np.sum(routing_weights_all * np.log(routing_weights_all + epsilon), axis=1)
mean_entropy = entropy_per_sample.mean()
max_entropy = np.log(num_experts)
specialization_score = 1 - (mean_entropy / max_entropy)

print(f"\nMean entropy (learned): {mean_entropy:.4f}")
print(f"Max entropy (random): {max_entropy:.4f}")
print(f"\n>>> SPECIALIZATION SCORE: {specialization_score:.1%}")

if specialization_score > 0.7:
    print("    ✓ STRONG SPECIALIZATION: Gating mechanism learned meaningful routing")
elif specialization_score > 0.4:
    print("    ◐ MODERATE SPECIALIZATION: Some expert specialization")
else:
    print("    ✗ WEAK SPECIALIZATION: Gating is nearly random")

# ============================================================================
# METRIC 2: ROUTING CONFIDENCE BY CLASS
# ============================================================================

print("\n" + "=" * 80)
print("METRIC 2: ROUTING CONFIDENCE BY CLASS")
print("=" * 80)

routing_confidence_by_class = []
for class_id in range(10):
    class_mask = all_labels == class_id
    class_weights = routing_weights_all[class_mask]
    confidence = class_weights.max(axis=1).mean()
    routing_confidence_by_class.append(confidence)

print("\nRouting Confidence per Class:")
print("-" * 80)
for class_id, class_name in enumerate(class_names):
    conf = routing_confidence_by_class[class_id]
    bar = "█" * int(conf * 40)
    print(f"{class_name:15s} | {bar:40s} {conf:.1%}")

avg_confidence = np.mean(routing_confidence_by_class)
print(f"\nAverage routing confidence: {avg_confidence:.1%}")

if avg_confidence > 0.6:
    print("✓ Routing shows CLEAR PREFERENCES for each class")
elif avg_confidence > 0.4:
    print("◐ Routing shows MODERATE preferences")
else:
    print("✗ Routing is very UNCERTAIN (close to uniform)")

# ============================================================================
# METRIC 3: EXPERT UTILIZATION & BALANCE
# ============================================================================

print("\n" + "=" * 80)
print("METRIC 3: EXPERT UTILIZATION BALANCE")
print("=" * 80)

expert_contributions = routing_weights_all.sum(axis=0)
total_contribution = expert_contributions.sum()
expert_utilization = expert_contributions / total_contribution

print(f"\nExpert Utilization:")
for expert_id in range(num_experts):
    util = expert_utilization[expert_id]
    bar = "█" * int(util * 40)
    print(f"  Expert {expert_id} | {bar:40s} {util:.1%}")

specialization_index = scipy_entropy(expert_utilization)
max_specialization = np.log(num_experts)
balance_score = 1 - (specialization_index / max_specialization)

print(f"\nUtilization Balance Score: {balance_score:.1%}")
if balance_score > 0.8:
    print("  ✓ HIGHLY BALANCED: Load balancing strongly enforced")
elif balance_score > 0.5:
    print("  ◐ MODERATELY BALANCED: Some specialization emerging")
else:
    print("  ✗ IMBALANCED: Experts are specializing on different classes")

# ============================================================================
# METRIC 4: CLASS-EXPERT AFFINITY
# ============================================================================

print("\n" + "=" * 80)
print("METRIC 4: WHAT DO EXPERTS SPECIALIZE IN?")
print("=" * 80)

specialization_matrix = np.zeros((10, num_experts))
for class_id in range(10):
    class_mask = all_labels == class_id
    class_routing_weights = routing_weights_all[class_mask]

    for expert_id in range(num_experts):
        total_contribution = class_routing_weights[:, expert_id].sum()
        specialization_matrix[class_id, expert_id] = total_contribution

specialization_pct = specialization_matrix / specialization_matrix.sum(axis=1, keepdims=True)

print("\nClass → Preferred Expert (Strongest Routing):")
print("-" * 80)
expert_class_mapping = {e: [] for e in range(num_experts)}

for class_id, class_name in enumerate(class_names):
    preferred_expert = specialization_pct[class_id].argmax()
    preference_strength = specialization_pct[class_id, preferred_expert]
    expert_class_mapping[preferred_expert].append((class_name, preference_strength))
    print(f"{class_name:15s} → Expert {preferred_expert} ({preference_strength:5.1%})")

print("\nExpert → Preferred Classes (Semantic Specialization):")
print("-" * 80)
for expert_id in range(num_experts):
    classes = expert_class_mapping[expert_id]
    if classes:
        classes_sorted = sorted(classes, key=lambda x: x[1], reverse=True)
        class_names_str = ", ".join([c[0] for c in classes_sorted])
        print(f"\nExpert {expert_id} specializes in:")
        print(f"  {class_names_str}")
    else:
        print(f"\nExpert {expert_id}: (no clear specialization)")

# ============================================================================
# METRIC 5: ROUTING CONSISTENCY
# ============================================================================

print("\n" + "=" * 80)
print("METRIC 5: ROUTING CONSISTENCY BY CLASS")
print("=" * 80)

routing_variance_by_class = []
for class_id in range(10):
    class_mask = all_labels == class_id
    class_weights = routing_weights_all[class_mask]
    preferred_expert_per_sample = class_weights.argmax(axis=1)
    variance = len(np.unique(preferred_expert_per_sample)) - 1
    routing_variance_by_class.append(variance)

print("\nRouting Consistency (0=all samples to one expert, 2=scattered across experts):")
print("-" * 80)
for class_id, class_name in enumerate(class_names):
    variance = routing_variance_by_class[class_id]
    consistency = "Consistent" if variance <= 1 else "Scattered"
    print(f"{class_name:15s} | {consistency:10s} (variance={variance})")

# ============================================================================
# SUMMARY
# ============================================================================

print("\n" + "=" * 80)
print("SUMMARY: DOES GATING WORK?")
print("=" * 80)

print(f"""
Based on the five metrics above, here's the answer:

1. ENTROPY SCORE ({specialization_score:.1%}):
   - Measures if gating learned to specialize
   - Your score: {"HIGH" if specialization_score > 0.7 else "MODERATE" if specialization_score > 0.4 else "LOW"}

2. ROUTING CONFIDENCE ({avg_confidence:.1%}):
   - Measures how certain routing is
   - Your score: {"HIGH" if avg_confidence > 0.6 else "MODERATE" if avg_confidence > 0.4 else "LOW"}

3. UTILIZATION BALANCE ({balance_score:.1%}):
   - Load balancing vs specialization trade-off
   - Your model: {"BALANCED" if balance_score > 0.8 else "MODERATE" if balance_score > 0.5 else "SPECIALIZED"}

4. SEMANTIC SPECIALIZATION:
   - {len([e for e in expert_class_mapping if expert_class_mapping[e]])} experts show clear specialization patterns

5. ROUTING CONSISTENCY:
   - {sum(1 for v in routing_variance_by_class if v <= 1)} out of 10 classes have consistent routing

OVERALL ASSESSMENT:
""")

if specialization_score > 0.6 and avg_confidence > 0.5:
    print("✓ GATING WORKS WELL: Mechanism is learning meaningful routing patterns")
elif specialization_score > 0.3:
    print("◐ GATING PARTIALLY WORKS: Some learning, but not strong specialization")
else:
    print("✗ GATING NOT EFFECTIVE: Routing is too uniform/random")

# ============================================================================
# VISUALIZATIONS
# ============================================================================

print("\n" + "=" * 80)
print("GENERATING VISUALIZATIONS...")
print("=" * 80)

fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Plot 1: Heatmap of class-expert affinity
sns.heatmap(specialization_pct * 100,
            xticklabels=[f'Expert {i}' for i in range(num_experts)],
            yticklabels=class_names,
            annot=True,
            fmt='.0f',
            cmap='YlOrRd',
            ax=axes[0, 0],
            cbar_kws={'label': 'Routing % by class'})
axes[0, 0].set_title('Class-Expert Affinity Matrix')
axes[0, 0].set_xlabel('Expert')
axes[0, 0].set_ylabel('Class')

# Plot 2: Routing confidence by class
axes[0, 1].barh(class_names, routing_confidence_by_class, color='steelblue')
axes[0, 1].axvline(avg_confidence, color='red', linestyle='--', label=f'Avg: {avg_confidence:.1%}')
axes[0, 1].set_xlabel('Confidence')
axes[0, 1].set_title('Routing Confidence by Class')
axes[0, 1].set_xlim([0, 1])
axes[0, 1].legend()

# Plot 3: Expert utilization
colors = ['#FF6B6B', '#4ECDC4', '#45B7D1']
axes[1, 0].bar([f'Expert {i}' for i in range(num_experts)],
               expert_utilization * 100,
               color=colors)
axes[1, 0].axhline(100/num_experts, color='red', linestyle='--', label='Perfect Balance')
axes[1, 0].set_ylabel('Utilization (%)')
axes[1, 0].set_title('Expert Utilization')
axes[1, 0].set_ylim([0, 100])
axes[1, 0].legend()

# Plot 4: Entropy histogram
axes[1, 1].hist(entropy_per_sample, bins=30, color='skyblue', edgecolor='black')
axes[1, 1].axvline(mean_entropy, color='red', linestyle='--',
                   label=f'Mean: {mean_entropy:.3f}')
axes[1, 1].axvline(max_entropy, color='orange', linestyle='--',
                   label=f'Random: {max_entropy:.3f}')
axes[1, 1].set_xlabel('Entropy per Sample')
axes[1, 1].set_ylabel('Frequency')
axes[1, 1].set_title('Distribution of Routing Entropy')
axes[1, 1].legend()

plt.tight_layout()
plt.savefig('gating_effectiveness_analysis.png', dpi=300, bbox_inches='tight')
print("✓ Saved: gating_effectiveness_analysis.png")
plt.close()

print("\n" + "=" * 80)
print("ANALYSIS COMPLETE!")
print("=" * 80)
