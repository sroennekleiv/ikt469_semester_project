from utils.dataset import Cifar100Dataset

if __name__ == '__main__':
    dataset = Cifar100Dataset(batch_size=64)
    train_df, test_df = dataset.get_dataloaders()
    class_names = dataset.get_class_names()

    # Autoencoder

    # Contrastive

    # Pretrained

    # Expert