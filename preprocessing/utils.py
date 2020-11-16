import torch
from torchvision.transforms import transforms

from preprocessing.style_transferATOR_dataset import StyleTransferATORDataset


def get_train_val_dataset(config):

    data_transforms = transforms.Compose([
        transforms.Resize(32),
        transforms.ToTensor()
    ])


    dataset_train = StyleTransferATORDataset(transform=data_transforms, max_size=config.preprocessing.dataset.training_size, index_offset=0, config=config)
    dataset_val = StyleTransferATORDataset(transform=data_transforms, max_size=config.preprocessing.dataset.validation_size, index_offset=config.preprocessing.dataset.training_size, config=config)

    training_generator = torch.utils.data.DataLoader(dataset_train, **config.preprocessing.dataloader)
    validation_generator = torch.utils.data.DataLoader(dataset_val, **config.preprocessing.dataloader)

    dataloaders = {
        'train': training_generator,
        'val': validation_generator
    }

    dataset_sizes = {
        'train': len(dataset_train),
        'val': len(dataset_val)
    }

    result_dict = dict()
    result_dict['dataloaders'] = dataloaders
    result_dict['dataset_sizes'] = dataset_sizes
    return result_dict

def get_train_val_dataset_AB(config):

    data_transforms = transforms.Compose([
        transforms.Resize(128),
        transforms.ToTensor()
    ])


    dataset_train_a = StyleTransferATORDataset(transform=data_transforms, max_size=config.preprocessing.dataset.training_size, index_offset=0, path='10K/trainA')
    dataset_val_a = StyleTransferATORDataset(transform=data_transforms, max_size=config.preprocessing.dataset.validation_size, index_offset=config.preprocessing.dataset.training_size, path='10K/trainA')
    dataset_train_b = StyleTransferATORDataset(transform=data_transforms, max_size=config.preprocessing.dataset.training_size, index_offset=0, path='10K/trainB')
    dataset_val_b = StyleTransferATORDataset(transform=data_transforms, max_size=config.preprocessing.dataset.validation_size, index_offset=config.preprocessing.dataset.training_size, path='10K/trainB')

    training_generator_a = torch.utils.data.DataLoader(dataset_train_a, **config.preprocessing.dataloader)
    validation_generator_a = torch.utils.data.DataLoader(dataset_val_a, **config.preprocessing.dataloader)

    training_generator_b = torch.utils.data.DataLoader(dataset_train_b, **config.preprocessing.dataloader)
    validation_generator_b = torch.utils.data.DataLoader(dataset_val_b, **config.preprocessing.dataloader)

    dataloaders = {
        'train_a': training_generator_a,
        'val_a': validation_generator_a,
        'train_b': training_generator_b,
        'val_b': validation_generator_b
    }

    dataset_sizes = {
        'train_a': len(dataset_train_a),
        'val_a': len(dataset_val_a),
        'train_b': len(dataset_train_b),
        'val_b': len(dataset_val_b)
    }

    result_dict = dict()
    result_dict['dataloaders'] = dataloaders
    result_dict['dataset_sizes'] = dataset_sizes
    return result_dict