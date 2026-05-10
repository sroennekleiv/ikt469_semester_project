import torch
import torch.nn.functional as F

# Orthoganility Loss function encourage the experts to learn different feature spaces
def orthogonality_loss(embeddings):
    num_experts = embeddings.size(1)

    loss = 0.0
    count = 0

    for i in range(num_experts):
        for j in range(i+1, num_experts):
            embedding_i = F.normalize(embeddings[:, i], dim=-1) # Get the embedding for expert i and normalize it
            embedding_j = F.normalize(embeddings[:, j], dim=-1) # Get the embedding for expert j and normalize it

            similarity = (embedding_i * embedding_j).sum(dim=-1).mean() # Compute the similarity between the embeddings (where lower values between them measn stronger specialization)
            loss += similarity.abs() # Encourage orthogonality by minimizing the absolute similarity
            count += 1

    return loss / max(count, 1)

# Routing Sharpnes Loss encourage confident routing decisions
def routing_sharpness_loss(gate_weights):
    max_probs = gate_weights.max(dim=1)[0] # Get the maximum routing weight for each sample
    return-max_probs.mean() # Encourage sharp routing by maximizing the average of the maximum routing weights

# Function to encourage consistency of expert embeddings for samples routed to the same expert, promoting specialization
def expert_consistency_loss(embeddings, gate_weights):
    dominant_experts = gate_weights.argmax(dim=1) # Get the index of the dominant expert for each sample
    num_experts = embeddings.size(1)

    loss = 0.0
    count = 0

    for expert in range(num_experts):
        mask = dominant_experts == expert # Mask to select samples routed to the current expert
        if mask.sum() < 2: # Continue if there are less than 2 samples for this expert to compute consistency
            continue

        selected_embeddings = embeddings[mask, expert] # Select the embeddings
        selected_embeddings = F.normalize(selected_embeddings, dim=-1)

        # Compute pairwise cosine similarity between embeddings routed to the same expert
        similarity_matrix = torch.matmul(selected_embeddings, selected_embeddings.T)

        mean_similarity = similarity_matrix.mean()

        loss += (1 - mean_similarity)
        count += 1
    
    if count == 0: # Return zero loss if no experts have enough samples to compute consistency
        return torch.tensor(0.0, device=embeddings.device)
    
    return loss / count

# Function to compute the entropy of the routing distribution, encouraging balanced expert usage and preventing routing from collapsing to a single expert
def routing_entropy_loss(gate_weights):
    entropy = -(gate_weights * torch.log(gate_weights + 1e-8)).sum(dim=-1).mean()
    return -entropy # Maximize entropy to encourage balanced routing across experts

# Function to calculate the balancing loss to prevent experts from dominating the routing decisions, promoting more equal usage of all experts
def loading_balancing_loss(gate_weights):
    expert_usage = gate_weights.mean(dim=0) # Get the expert usage by averaging the routing weights across the batch
    uniform = torch.ones_like(expert_usage) / expert_usage.size(0)

    # Use KL divergence to encourage the expert usage distribution to be close to uniform, promoting balanced expert utilization
    loss = F.kl_div(torch.log(expert_usage + 1e-8), uniform, reduction="batchmean")
    return loss