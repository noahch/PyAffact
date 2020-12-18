import os

import pandas as pd
import torch
from torchvision.transforms import transforms

from utils.config_utils import save_config_to_file
from preprocessing.affact_transformer import AffactTransformer
from preprocessing.celeb_a_dataset import CelebADataset


def get_train_val_dataset(config):
    bounding_boxes, labels, landmarks, partition_df = _load_dataframes(config)

    df_train_labels, df_train_landmarks, df_train_bounding_boxes = _get_partition_dataframes(partition_df,
                                                                                             0,
                                                                                             labels,
                                                                                             landmarks,
                                                                                             bounding_boxes)

    df_val_labels, df_val_landmarks, df_val_bounding_boxes = _get_partition_dataframes(partition_df,
                                                                                       1,
                                                                                       labels,
                                                                                       landmarks,
                                                                                       bounding_boxes)

    data_transforms = transforms.Compose([AffactTransformer(config)])

    dataset_train, training_generator = generate_dataset_and_loader(data_transforms, df_train_labels,
                                                                    df_train_landmarks, df_train_bounding_boxes, config)

    dataset_val, validation_generator = generate_dataset_and_loader(data_transforms, df_val_labels, df_val_landmarks,
                                                                    df_val_bounding_boxes, config)

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
    train_attribute_baseline_majority_value.to_pickle(os.path.join(config.basic.result_directory, 'train_majority.pkl'),
                                                      compression='zip')
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


def _load_dataframes(config):
    labels = pd.read_csv(config.preprocessing.dataset.dataset_labels_filename,
                         delim_whitespace=True, skiprows=1)
    landmarks = pd.read_csv(config.preprocessing.dataset.landmarks_filename,
                            delim_whitespace=True, skiprows=1)
    bounding_boxes = pd.read_csv(config.preprocessing.dataset.bounding_boxes_filename,
                                 delim_whitespace=True, skiprows=1)
    partition_df = pd.read_csv(config.preprocessing.dataset.partition_filename,
                               delim_whitespace=True, header=None)

    return bounding_boxes, labels, landmarks, partition_df


def _get_partition_dataframes(partition_df, partition, labels, landmarks, bounding_boxes):
    partition_df.columns = ['filename', 'partition']
    partition_df["partition"] = pd.to_numeric(partition_df["partition"])

    train_partition_df = partition_df.loc[partition_df["partition"] == partition]
    df_labels = pd.merge(labels, train_partition_df, left_index=True, right_on='filename', how='inner')
    df_labels.set_index('filename', inplace=True)

    df_labels.drop(columns=["partition"], inplace=True)
    df_landmarks = pd.merge(landmarks, train_partition_df, left_index=True, right_on='filename', how='inner')
    df_landmarks.set_index('filename', inplace=True)
    df_landmarks.drop(columns=["partition"], inplace=True)
    df_bounding_boxes = pd.merge(bounding_boxes, train_partition_df, left_on='image_id', right_on='filename',
                                 how='inner')
    df_bounding_boxes.drop(columns=["filename", "partition"], inplace=True)
    return df_labels, df_landmarks, df_bounding_boxes


def generate_dataset_and_loader(transform, labels, landmarks, bounding_boxes, config):
    dataset = CelebADataset(transform=transform, labels=labels, landmarks=landmarks, bounding_boxes=bounding_boxes,
                            config=config)
    dataloader = torch.utils.data.DataLoader(dataset, **config.preprocessing.dataloader)
    return dataset, dataloader
