import torch
from torchvision.transforms import transforms
import pandas as pd
import numpy as np

from config.config_utils import save_config_to_file
from preprocessing.affact_dataset import AffactDataset
from preprocessing.affact_transformer import AffactTransformer
import os


def get_train_val_dataset(config):
    if config.preprocessing.transformation.use_affact_transformator:
        data_transforms = transforms.Compose([AffactTransformer(config)])
    else:
        # Use other transformation than affact
        raise NotImplemented("Implement your own Transformer here")

    assert config.preprocessing.dataset.uses_bounding_boxes != config.preprocessing.dataset.uses_landmarks, "Either use landmarks or bounding boxes"

    labels = pd.read_csv(config.preprocessing.dataset.dataset_labels_filename,
                         delim_whitespace=True, skiprows=1)
    idx = np.random.permutation(labels.index)
    # TODO: uncomment
    labels = labels.reindex(idx)
    landmarks = None
    if config.preprocessing.dataset.uses_landmarks:
        landmarks = pd.read_csv(config.preprocessing.dataset.landmarks_filename,
                                delim_whitespace=True, skiprows=1)

        assert labels.shape[0] == landmarks.shape[0], "Label and Landmarks not of same shape"
        # TODO: uncomment
        landmarks = landmarks.reindex(idx)

    bounding_boxes = None
    if config.preprocessing.dataset.uses_bounding_boxes:
        bounding_boxes = pd.read_csv(config.preprocessing.dataset.bounding_boxes_filename,
                                delim_whitespace=True, skiprows=1)

        assert labels.shape[0] == bounding_boxes.shape[0], "Label and bounding boxes not of same shape"
        # TODO: uncomment
        bounding_boxes = bounding_boxes.reindex(idx)

    # If number of samples is -1 --> use whole dataset
    if config.preprocessing.dataset.number_of_samples == -1:
        size = labels.shape[0]
    else:
        size = config.preprocessing.dataset.number_of_samples

    assert config.preprocessing.dataset.train_fraction + config.preprocessing.dataset.val_fraction + config.preprocessing.dataset.test_fraction == 1, "Train/Val/Test split must sum up to 1"

    df_train_labels, df_val_labels, df_test_labels = calculate_split(config, labels, size)
    df_train_labels.to_pickle('{}/train_labels.pkl'.format(config.basic.result_directory), compression='zip')
    df_val_labels.to_pickle('{}/val_labels.pkl'.format(config.basic.result_directory), compression='zip')
    df_test_labels.to_pickle(os.path.join(config.basic.result_directory, 'test_labels.pkl'), compression='zip')
    config.evaluation.test_labels_pickle_filename = 'test_labels.pkl'

    df_train_landmarks, df_val_landmarks, df_test_landmarks = None, None, None
    if config.preprocessing.dataset.uses_landmarks:
        df_train_landmarks, df_val_landmarks, df_test_landmarks = calculate_split(config, landmarks, size)
        df_train_landmarks.to_pickle('{}/train_landmarks.pkl'.format(config.basic.result_directory), compression='zip')
        df_val_landmarks.to_pickle('{}/val_landmarks.pkl'.format(config.basic.result_directory), compression='zip')
        df_test_landmarks.to_pickle(os.path.join(config.basic.result_directory, 'test_landmarks.pkl'), compression='zip')

        config.evaluation.test_landmarks_pickle_filename = 'test_landmarks.pkl'

    df_train_bounding_boxes, df_val_bounding_boxes, df_test_bounding_boxes = None, None, None
    if config.preprocessing.dataset.uses_bounding_boxes:
        df_train_bounding_boxes, df_val_bounding_boxes, df_test_bounding_boxes = calculate_split(config, bounding_boxes, size)
        df_train_bounding_boxes.to_pickle('{}/train_bounding_boxes.pkl'.format(config.basic.result_directory), compression='zip')
        df_val_bounding_boxes.to_pickle('{}/val_bounding_boxes.pkl'.format(config.basic.result_directory), compression='zip')
        df_test_bounding_boxes.to_pickle(os.path.join(config.basic.result_directory, 'test_bounding_boxes.pkl'),
                                    compression='zip')

        config.evaluation.test_bounding_boxes = 'test_bounding_boxes.pkl'

    dataset_train, training_generator = generate_dataset_and_loader(data_transforms, df_train_labels, df_train_landmarks, df_train_bounding_boxes, config)
    dataset_val, validation_generator = generate_dataset_and_loader(data_transforms, df_val_labels, df_val_landmarks, df_val_bounding_boxes, config)

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
    config.evaluation.train_majority_pickle_filename = 'train_majority.pkl'

    # Save for evaluation
    train_attribute_baseline_majority_value.to_pickle(os.path.join(config.basic.result_directory, 'train_majority.pkl'), compression='zip')
    save_config_to_file(config)

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

def generate_dataset_and_loader(transform, labels, landmarks, bounding_boxes, config):
    dataset = AffactDataset(transform=transform, labels=labels, landmarks=landmarks, bounding_boxes=bounding_boxes,
                  config=config)
    if config.basic.mode == 'train':
        dataloader = torch.utils.data.DataLoader(dataset, **config.preprocessing.dataloader)
    else:
        sampler = torch.utils.data.SequentialSampler(dataset)
        dataloader = torch.utils.data.DataLoader(dataset, **config.evaluation.dataloader, sampler=sampler)

    return dataset, dataloader

def calculate_split(config, df, size):
    df_train = df[0: int(size * config.preprocessing.dataset.train_fraction)]
    df_val = df[int(size * config.preprocessing.dataset.train_fraction): int(
        size * (config.preprocessing.dataset.val_fraction + config.preprocessing.dataset.train_fraction))]
    df_test = df[int(size * (config.preprocessing.dataset.val_fraction + config.preprocessing.dataset.train_fraction)): size]
    return df_train, df_val, df_test
