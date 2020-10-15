import torch
from torchvision.transforms import transforms

from preprocessing.affact_dataset import AffactDataset
from preprocessing.affact_transformer import AffactTransformer


def get_train_val_dataset(config):
    if config.preprocessing.dataset.pre_aligned:
        data_transforms = transforms.Compose([
            transforms.Resize(224),
            transforms.ToTensor()
        ])
    else:
        data_transforms = transforms.Compose([
            transforms.Resize(224),
            transforms.ToTensor()
        ])

    data_transforms = transforms.Compose([AffactTransformer(config)])

    dataset_train = AffactDataset(transform=data_transforms, max_size=config.preprocessing.dataset.training_size, index_offset=0, config=config)
    dataset_val = AffactDataset(transform=data_transforms, max_size=config.preprocessing.dataset.validation_size, index_offset=config.preprocessing.dataset.training_size, config=config)

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