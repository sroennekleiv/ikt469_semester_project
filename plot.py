import matplotlib.pyplot as plt
import numpy as np

# Fashion-MNIST classes
classes = [
    "T-shirt", "Trouser", "Pullover", "Dress", "Coat",
    "Sandal", "Shirt", "Sneaker", "Bag", "Ankle Boot"
]

# Intra-class distances
autoencoder = [0.9481, 0.8329, 0.9741, 0.9452, 0.9323,
               0.7593, 0.9785, 0.8048, 1.0117, 0.8832]

contrastive = [0.0200, 0.0102, 0.0213, 0.0161, 0.0178,
               0.0109, 0.0219, 0.0087, 0.0210, 0.0137]

clip = [0.3921, 0.3128, 0.3581, 0.3998, 0.3616,
        0.4178, 0.3971, 0.3904, 0.4300, 0.3400]

moe = [0.3223, 0.2025, 0.4165, 0.2762, 0.3517,
       0.1615, 0.3927, 0.1358, 0.3197, 0.2037]

x = np.arange(len(classes))
width = 0.2

# Plot
plt.figure(figsize=(18, 8))

plt.bar(x - 1.5 * width, autoencoder, width, label='Autoencoder')
plt.bar(x - 0.5 * width, contrastive, width, label='Contrastive')
plt.bar(x + 0.5 * width, clip, width, label='CLIP')
plt.bar(x + 1.5 * width, moe, width, label='MoE')

# Labels and formatting
plt.xticks(x, classes, rotation=20, fontsize=14)
plt.yticks(fontsize=14)

plt.xlabel("Fashion-MNIST Class", fontsize=18)
plt.ylabel("Intra-class Distance", fontsize=18)

plt.title(
    "Per-class Intra-class Distances Across Representation Learning Models",
    fontsize=22,
    pad=20
)

plt.legend(fontsize=14)

plt.grid(axis='y')

plt.tight_layout()
plt.savefig("intra_class_distances.png", dpi=300)