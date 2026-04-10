import torch

from utils.dataset import Cifar100Dataset

from training_and_evaluate.train_expert import ExpertTrainerAndEvaluator
from training_and_evaluate.train_autoencoder import AutoEncoderTrainerAndEvaluator
from training_and_evaluate.train_contrastive import ContrastiveTrainerAndEvaluator

from training_and_evaluate.extract_embeddings import ExtractEmbeddings, VisualizeEmbeddings

from models.mixtureofexperts.mixture_of_experts import MixtureOfExperts
from models.contrastive import ContrastiveModel
from models.pretrained import PretrainedModel
from models.autoencoder import AutoencoderModel

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

EPOCHS = 30

if __name__ == '__main__':
    dataset = Cifar100Dataset(batch_size=128, num_workers=1)
    train_df, test_df = dataset.get_dataloaders(subset_size=1000)
    class_names = dataset.get_class_names()

    print(f"Number of training samples: {len(train_df.dataset)}")
    print(f"Number of test samples: {len(test_df.dataset)}")
    print(f"Class names length: {len(class_names)}")

    NUM_CLASSES = len(class_names)

    embedding_extractor = ExtractEmbeddings(device=device)
    visualizer = VisualizeEmbeddings()

    '''# Autoencoder
    autoencoder_model = AutoencoderModel(embedding_dim=128).to(device)
    autoencoder_trainer = AutoEncoderTrainerAndEvaluator(autoencoder_model)
    test_loss = autoencoder_trainer.run_experiment(train_df, test_df, epochs=EPOCHS)
    embeddings, labels, images = embedding_extractor.extract_embeddings(autoencoder_model, test_df, num_classes=NUM_CLASSES)
    visualizer.visualize(model_name='autoencoder', embeddings=embeddings, labels=labels)'''

    # Contrastive
    contrastive_model = ContrastiveModel(in_channels=3, embedding_dim=128, projection_dim=64).to(device)
    contrastive_trainer = ContrastiveTrainerAndEvaluator(contrastive_model, margin=1.0, device=device)
    test_loss, test_accuracy = contrastive_trainer.run_experiment(train_df, test_df, epochs=EPOCHS)
    embeddings, labels, images = embedding_extractor.extract_embeddings(contrastive_model, test_df, num_classes=NUM_CLASSES)
    visualizer.visualize(model_name='contrastive', embeddings=embeddings, labels=labels)

    '''# Pretrained

    # Experts
    mixture_of_experts = MixtureOfExperts(num_experts=3, num_classes=len(class_names))
    expert_trainer = ExpertTrainerAndEvaluator(mixture_of_experts, device)
    test_loss, test_acc = expert_trainer.run_experiment(train_df, test_df, epochs=EPOCHS)
    '''
