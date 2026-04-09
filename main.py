import torch

from utils.dataset import Cifar100Dataset

from training_and_evaluate.train_expert import ExpertTrainerAndEvaluator
from training_and_evaluate.train_contrastive import ContrastiveTrainerAndEvaluator

from models.mixtureofexperts.mixture_of_experts import MixtureOfExperts
from models.contrastive import ContrastiveModel
from models.pretrained import PretrainedModel
from models.autoencoder import AutoencoderModel

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

EPOCHS = 10

if __name__ == '__main__':
    dataset = Cifar100Dataset(batch_size=64)
    train_df, test_df = dataset.get_dataloaders()
    class_names = dataset.get_class_names()

    print(f"Number of training samples: {len(train_df.dataset)}")
    print(f"Number of test samples: {len(test_df.dataset)}")
    print(f"Class names: {class_names}")

    # Autoencoder

    # Contrastive
    contrastive_model = ContrastiveModel(embedding_dim=128, projection_dim=64).to(device)
    contrastive_trainer = ContrastiveTrainerAndEvaluator(contrastive_model, margin=1.0, device=device)
    test_loss, test_acc = contrastive_trainer.run_experiment(train_df, test_df, epochs=EPOCHS)

    # Pretrained

    '''# Expert
    mixture_of_experts = MixtureOfExperts(num_experts=3, num_classes=len(class_names))
    expert_trainer = ExpertTrainerAndEvaluator(mixture_of_experts, device)
    test_loss, test_acc = expert_trainer.run_experiment(train_df, test_df, epochs=EPOCHS)
    '''
