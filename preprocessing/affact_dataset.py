import torch
import pandas as pd
import numpy as np
import bob.io.image
from PIL import Image
import logging
from torchvision.transforms import transforms

from evaluation.utils import save_input_transform_output_image


class AffactDataset(torch.utils.data.Dataset):
    def __init__(self, transform=None, max_size=None, index_offset=0, config=None):
        'Initialization'
         # TODO: Factor for multiplication of input
        
        self.labels = pd.read_csv('dataset/{}'.format(config.preprocessing.dataset.dataset_labels_filename), delim_whitespace=True)
        # self.bboxes = pd.read_csv('dataset/CelebA/list_bbox_celeba.txt', delim_whitespace=True)
        if config.preprocessing.dataset.uses_landmarks:
            self.landmarks = pd.read_csv('dataset/{}'.format(config.preprocessing.dataset.landmarks_filename), delim_whitespace=True)
        else:
            # TODO: Manually detect landmarks (eyes and mouth)
            raise Exception('Manual Landmark detection not yet implemented')

        if max_size:
            self.labels = self.labels.iloc[index_offset:index_offset+max_size]
            self.landmarks = self.landmarks.iloc[index_offset:index_offset+max_size]
        self.transform = transform
        assert self.transform is not None, "A basic transformation is needed. i.e.  Resize() and ToTensor()"
        self.config = config


    def __len__(self):
        'Denotes the total number of samples'
        return self.labels.shape[0]


    def __getitem__(self, index):
        'Generates one sample of data'
        x = self.labels.iloc[index].name
        y = np.array(self.labels.iloc[index].array)
        y = np.where(y<0, 0, y)

        if 'AffactTransformer' in '{}'.format(self.transform):
            # Load data and get label
            image = bob.io.base.load('dataset/{}/{}'.format(self.config.preprocessing.dataset.dataset_image_folder, x))
            landmarks = self.landmarks.iloc[index].tolist()
            landmarks = landmarks[:4] + landmarks[6:]
            input = {
                'image': image,
                'landmarks': landmarks,
                'index': index
            }
            X, bbx = self.transform(input)
        else:
            image = Image.open('dataset/{}/{}'.format(self.config.preprocessing.dataset.dataset_image_folder, x))
            X = self.transform(image)
            bbx = None

        # Save every X picture to validate preprocessing
        if self.config.preprocessing.transformation.save_transformation_image.enabled:
            if index % self.config.preprocessing.transformation.save_transformation_image.frequency == 0:
                save_input_transform_output_image(index, image, X, self.config.basic.result_directory, bbx)


        return X, y, index

    def get_attribute_baseline_accuracy(self):
        return self.labels.apply(pd.Series.value_counts).max() / len(self.labels)

    def get_label_names(self):
        return self.labels.columns