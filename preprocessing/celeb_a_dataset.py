import torch
import pandas as pd
import numpy as np
import bob.io.image

from facenet_pytorch.models.mtcnn import MTCNN

from evaluation.utils import save_original_and_preprocessed_image


class CelebADataset(torch.utils.data.Dataset):
    """
    dataset class for the CelebA dataset
    """
    def __init__(self, transform=None, labels=None, landmarks=None, bounding_boxes=None, config=None):
        """
        init
        :param transform: transformations to apply to each image
        :param labels: ground truth dataframe (y-label)
        :param landmarks: landmarks dataframe
        :param bounding_boxes: bounding box dataframe
        :param config: training configuration file
        """

        self.labels = labels

        # use landmarks
        if config.dataset.bounding_box_mode == 0:
            self.landmarks = landmarks

        # use bounding boxes
        elif config.dataset.bounding_box_mode == 1:
            self.bounding_boxes = bounding_boxes

        # use face detector
        elif config.dataset.bounding_box_mode == 2:
            self.mtcnn = MTCNN(select_largest=False, device=config.basic.cuda_device_name.split(',')[0])

        else:
            raise Exception("Chose a valid bounding_box_mode (0=landmarks hand-labeled, 1=bbx hand-labeled, 2=bbx detected")

        self.transform = transform
        assert self.transform is not None, "A basic transformation is needed. i.e.  Resize() and ToTensor()"
        self.config = config


    def __len__(self):
        """Denotes the total number of samples"""
        return self.labels.shape[0]


    def __getitem__(self, index):
        """
        Processes one sample
        :param index: sample index
        :return: X (Tensor of image), y (label), index (sample index)
        """

        # Get image name
        image_name = self.labels.iloc[index].name

        # Get label
        y = np.array(self.labels.iloc[index].array)

        # set -1 to 0 for model
        y = np.where(y<0, 0, y)

        # Load data and get label
        image = bob.io.base.load('{}/{}'.format(self.config.dataset.dataset_image_folder, image_name))

        # Prepare bounding boxes/landmarks for transformer
        landmarks, bounding_boxes = None, None
        if self.config.dataset.bounding_box_mode == 0:
            landmarks = self.landmarks.iloc[index].tolist()
            landmarks = landmarks[:4] + landmarks[6:]
        elif self.config.dataset.bounding_box_mode == 1:
            bounding_boxes = self.bounding_boxes.iloc[index].tolist()
            bounding_boxes = bounding_boxes[1:]
            if self.config.dataset.bounding_box_scale:
                scale = self.config.dataset.bounding_box_scale
                bounding_boxes[0] = bounding_boxes[0] - ((scale - 1) / 2 * bounding_boxes[2])
                bounding_boxes[1] = bounding_boxes[1] - ((scale - 1) / 2 * bounding_boxes[3])
                bounding_boxes[2] = scale * (bounding_boxes[2])
                bounding_boxes[3] = scale * (bounding_boxes[3])

        # Create input structure
        input = {
            'image': image,
            'landmarks': landmarks,
            'bounding_boxes': bounding_boxes,
            'index': index
        }

        # Apply AFFACT transformer
        X = self.transform(input)

        # Save every X picture to validate preprocessing
        if self.config.preprocessing.save_preprocessed_image.enabled:
            if index % self.config.preprocessing.save_preprocessed_image.frequency == 0:
                save_original_and_preprocessed_image(index, image, X, self.config.basic.result_directory)

        return X, y, index

    def get_attribute_baseline_accuracy(self):
        """
        Calculate baseline accuracy on training data
        :return: baseline accuracy (for training set)
        """
        return self.labels.apply(pd.Series.value_counts).max() / len(self.labels)

    def get_attribute_baseline_majority_value(self):
        """
        Get majority value for each attribute
        :return: list of majority values
        """
        return self.labels.apply(pd.Series.value_counts).idxmax()

    def get_attribute_baseline_accuracy_val(self, train_attribute_baseline_majority_value):
        """
        Calculate baseline accuracy on validation data based on majority value extracted from training data
        :param train_attribute_baseline_majority_value: majority value of training data
        :return: baseline accuracy (for validation set)
        """
        x = self.labels.apply(pd.Series.value_counts)
        lst = train_attribute_baseline_majority_value.tolist()
        access_tuple_list = [(0 if lst[y] == -1 else 1, y) for y in range(0, len(lst))]
        result_list = []
        for t in access_tuple_list:
            result_list.append((train_attribute_baseline_majority_value.keys()[t[1]], x.iloc[t] / self.labels.shape[0]))
        return pd.DataFrame(result_list).set_index(0)[1]

    def get_label_names(self):
        """
        Get label names
        :return: label names
        """
        return self.labels.columns