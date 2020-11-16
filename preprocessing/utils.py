import torch
from torchvision.transforms import transforms

from preprocessing.affact_dataset import AffactDataset
from preprocessing.affact_transformer import AffactTransformer


def get_train_val_dataset(config):
    if config.preprocessing.transformation.use_affact_transformator:
        data_transforms = transforms.Compose([AffactTransformer(config)])
    else:
        # Use other transformation than affact
        data_transforms = transforms.Compose([
            transforms.CenterCrop([224, 224]),
            # transforms.Resize([224, 224]),
            transforms.ToTensor()
        ])

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

    # We use the attribute distribution of the train dataset for the validation data, since Y would not be known.
    train_attribute_baseline_majority_value = dataset_train.get_attribute_baseline_majority_value()
    attribute_baseline_accuracy = {
        'train': dataset_train.get_attribute_baseline_accuracy(),
        'val': dataset_val.get_attribute_baseline_accuracy_val(train_attribute_baseline_majority_value)
    }

    dataset_meta_information = {
        'label_names': dataset_train.get_label_names(),
        'number_of_labels': len(dataset_train.get_label_names())
    }

    result_dict = dict()
    result_dict['dataloaders'] = dataloaders
    result_dict['dataset_sizes'] = dataset_sizes
    result_dict['attribute_baseline_accuracy'] = attribute_baseline_accuracy
    result_dict['dataset_meta_information'] = dataset_meta_information
    return result_dict