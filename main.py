import torch
import time

from torchvision import transforms

from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import torch.nn.functional as F
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

from utils.dataset import FashionMNISTDataset

from training_and_evaluate.train_expert import ExpertTrainerAndEvaluator
from training_and_evaluate.train_autoencoder import AutoEncoderTrainerAndEvaluator
from training_and_evaluate.train_contrastive import ContrastiveTrainerAndEvaluator

from utils.benchmark import ExperimentBenchmarker, VisualizeEmbeddings

from models.mixtureofexperts.mixture_of_experts import MixtureOfExperts
from models.contrastive import ContrastiveModel
from models.pretrained import PretrainedCLIPModel
from models.autoencoder import AutoencoderModel

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

EPOCHS = 20

if __name__ == '__main__':
    dataset = FashionMNISTDataset(batch_size=64, size=32
                                  , split_fraction=0.5)
    train_df, test_df = dataset.get_dataloaders(is_contrastive=False) # Enable and disable preprocessing on the dataset
    class_names = dataset.get_class_names()

    print(f"Train: {len(train_df.dataset)}, Test: {len(test_df.dataset)}")

    NUM_CLASSES = len(class_names)

    benchmarker = ExperimentBenchmarker(device=device)
    visualizer = VisualizeEmbeddings()

    '''# Autoencoder
    autoencoder_model = AutoencoderModel(in_channels=1, embedding_dim=128).to(device)
    autoencoder_trainer = AutoEncoderTrainerAndEvaluator(autoencoder_model)
    test_loss = autoencoder_trainer.run_experiment(train_df, test_df, epochs=EPOCHS)

    start_time = time.time()
    auto_embeddings, auto_labels, auto_images = benchmarker.extract_embeddings(autoencoder_model, test_df, num_classes=NUM_CLASSES)
    end_time = time.time()

    visualizer.visualize(model_name='autoencoder', embeddings=auto_embeddings, labels=auto_labels)
    elapsed_time = end_time - start_time
    print(f"Time to extract embeddings: {elapsed_time:.2f} seconds")

    auto_results = benchmarker.run_experiment_functions(embeddings=auto_embeddings, labels=auto_labels)
    print(f'Auto encoder Results: {auto_results}')

    
    # Pretrained CLIP model with ViT-B/32 architecture
    clip_model = PretrainedCLIPModel(device=device)
    start_time = time.time()
    clip_embeddings, clip_labels, clip_images = benchmarker.extract_embeddings(clip_model, test_df, num_classes=NUM_CLASSES)
    end_time = time.time()
    
    visualizer.visualize(model_name='CLIP', embeddings=clip_embeddings, labels=clip_labels)
    elapsed_time = end_time - start_time
    print(f"Time to extract embeddings: {elapsed_time:.2f} seconds")

    clip_results = benchmarker.run_experiment_functions(embeddings=clip_embeddings, labels=clip_labels)
    print(f'CLIP Results: {clip_results}')'''
    

    # Experts with gating mechanism
    mixture_of_experts = MixtureOfExperts(num_classes=len(class_names))
    expert_trainer = ExpertTrainerAndEvaluator(mixture_of_experts, device)
    test_loss, test_acc = expert_trainer.run_experiment(train_df, test_df, epochs=EPOCHS)
    
    start_time = time.time()
    moe_embeddings, moe_labels, moe_images = benchmarker.extract_embeddings(mixture_of_experts, test_df, num_classes=NUM_CLASSES)
    end_time = time.time()
    visualizer.visualize(model_name='mixtureofexperts', embeddings=moe_embeddings, labels=moe_labels)
    elapsed_time = end_time - start_time
    print(f"Time to extract embeddings: {elapsed_time:.2f} seconds")

    moe_results = benchmarker.run_experiment_functions(embeddings=moe_embeddings, labels=moe_labels)
    print(f'MoE Results: {moe_results}')

    '''
    # Contrastive
    contrastive_model = ContrastiveModel(in_channels=1, embedding_dim=128, projection_dim=64).to(device)
    contrastive_trainer = ContrastiveTrainerAndEvaluator(contrastive_model, temperature=0.07, device=device)
    test_loss = contrastive_trainer.run_experiment(train_df, test_df, epochs=EPOCHS)
    
    start_time = time.time()
    embeddings, labels, images = benchmarker.extract_embeddings(contrastive_model, test_df, num_classes=NUM_CLASSES)
    end_time = time.time()
    visualizer.visualize(model_name='contrastive', embeddings=embeddings, labels=labels)
    elapsed_time = end_time - start_time
    print(f"Time to extract embeddings: {elapsed_time:.2f} seconds")

    contrastive_results = benchmarker.run_experiment_functions(embeddings=embeddings, labels=labels)
    print(f'Contrastive Results: {contrastive_results}')'''
    
