"""
function to generate the training and validation dataset
"""
import os
import pandas as pd
import torch
from torchvision.transforms import transforms
from preprocessing.affact_transformer import AffactTransformer
from preprocessing.celeb_a_dataset import CelebADataset


def get_train_val_dataset(config):
    """
    generate dataloader, dataset and all meta information needed for training
    Parameters
    ----------
    config the configuration file

    Returns
    -------
    result dict containing dataloader, dataset size, attribute baseline accuracy, dataset meta information
    """
    # Loads the bounding boxes, labels, landmarks and partition df from disk
    bounding_boxes, labels, landmarks, partition_df = _load_dataframes(config)

    # Gets the training labels, landmarks, bounding boxes according to the partition file
    df_train_labels, df_train_landmarks, df_train_bounding_boxes = _get_partition_dataframes(partition_df,
                                                                                             0,
                                                                                             labels,
                                                                                             landmarks,
                                                                                             bounding_boxes)

    # Gets the validation labels, landmarks, bounding boxes according to the partition file
    df_val_labels, df_val_landmarks, df_val_bounding_boxes = _get_partition_dataframes(partition_df,
                                                                                       1,
                                                                                       labels,
                                                                                       landmarks,
                                                                                       bounding_boxes)

    # Define the transformations that are applied to each image
    data_transforms = transforms.Compose([AffactTransformer(config)])

    # Generates the data for training
    dataset_train, training_generator = generate_dataset_and_loader(data_transforms, df_train_labels,
                                                                    df_train_landmarks, df_train_bounding_boxes, config)

    # Generates the data for validation
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

    # Save for evaluation
    train_attribute_baseline_majority_value.to_pickle(os.path.join(config.basic.result_directory, 'train_majority.pkl'),
                                                      compression='zip')

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
    """
    loads the labels, landmarks, bounding boxes and partition dataframe from disk

    Parameters
    ----------
    config the configuration file

    Returns
    -------
    bounding_boxes, labels, landmarks, partition dataframe
    """
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
    """
    generate labels, landmarks and bounding boxes dataframes based on partition (train/val/test)

    Parameters
    ----------
    partition_df partition dataframe
    partition 0=train, 1=val, 2=test
    labels labels dataframe
    landmarks landmarks dataframe
    bounding_boxes bounding boxes dataframe

    Returns
    -------
    labels dataframe, landmarks dataframe, bounding boxes dataframe
    """
    # Rename columns
    partition_df.columns = ['filename', 'partition']

    # Partition is a number
    partition_df["partition"] = pd.to_numeric(partition_df["partition"])

    # Filter on partition
    filtered_partition_df = partition_df.loc[partition_df["partition"] == partition]

    # Merge original labels with filtered partition dataframe
    df_labels = pd.merge(labels, filtered_partition_df, left_index=True, right_on='filename', how='inner')
    df_labels.set_index('filename', inplace=True)
    df_labels.drop(columns=["partition"], inplace=True)

    # Merge original landmarks with filtered partition dataframe
    df_landmarks = pd.merge(landmarks, filtered_partition_df, left_index=True, right_on='filename', how='inner')
    df_landmarks.set_index('filename', inplace=True)
    df_landmarks.drop(columns=["partition"], inplace=True)

    # Merge original bounding boxes with filtered partition dataframe
    df_bounding_boxes = pd.merge(bounding_boxes, filtered_partition_df, left_on='image_id', right_on='filename',
                                 how='inner')
    df_bounding_boxes.drop(columns=["filename", "partition"], inplace=True)
    return df_labels, df_landmarks, df_bounding_boxes


def generate_dataset_and_loader(transform, labels, landmarks, bounding_boxes, config):
    """
    creates a dataset instance of the CelebAdataset given the transformer, labels, landmarks, bounding boxes and the configuration file

    Parameters
    ----------
    transform AffactTransformer
    labels labels
    landmarks landmarks
    bounding_boxes bounding box
    config the configuration file

    Returns
    -------
    the dataset and dataloader
    """
    dataset = CelebADataset(transform=transform, labels=labels, landmarks=landmarks, bounding_boxes=bounding_boxes,
                            config=config)
    dataloader = torch.utils.data.DataLoader(dataset, **config.preprocessing.dataloader)
    return dataset, dataloader
