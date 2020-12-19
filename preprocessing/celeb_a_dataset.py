import torch
import pandas as pd
import numpy as np
import bob.io.image
from PIL import Image
import logging

from facenet_pytorch.models.mtcnn import MTCNN
from torchvision.transforms import transforms

from evaluation.utils import save_input_transform_output_image, save_image


class CelebADataset(torch.utils.data.Dataset):
    def __init__(self, transform=None, labels=None, landmarks=None, bounding_boxes=None, config=None):
        'Initialization'

        self.labels = labels
        if config.preprocessing.dataset.bounding_box_mode == 0:
            self.landmarks = landmarks
        elif config.preprocessing.dataset.bounding_box_mode == 1:
            self.bounding_boxes = bounding_boxes
        elif config.preprocessing.dataset.bounding_box_mode == 2:
            self.mtcnn = MTCNN(select_largest=False, device=config.basic.cuda_device_name.split(',')[0])
        else:
            raise Exception("Chose a valid bounding_box_mode (0=landmarks hand-labeled, 1=bbx hand-labeled, 2=bbx detected")


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

        # If the AffactTransformer is used, the input format required changes (also includes landmarks, and index)
        if 'AffactTransformer' in '{}'.format(self.transform):
            # Load data and get label
            image = bob.io.base.load('{}/{}'.format(self.config.preprocessing.dataset.dataset_image_folder, x))
            landmarks, bounding_boxes = None, None
            if self.config.preprocessing.dataset.bounding_box_mode == 0:
                landmarks = self.landmarks.iloc[index].tolist()
                landmarks = landmarks[:4] + landmarks[6:]
            elif self.config.preprocessing.dataset.bounding_box_mode == 1:
                bounding_boxes = self.bounding_boxes.iloc[index].tolist()
                bounding_boxes = bounding_boxes[1:]
            elif self.config.preprocessing.dataset.bounding_box_mode == 2:
                boxes, probs, lm = self.mtcnn.detect(Image.fromarray(np.transpose(image, (1, 2, 0)), 'RGB'), landmarks=True)
                landmarks = [lm[0][0][0], lm[0][0][1], lm[0][1][0], lm[0][1][1],
                             lm[0][3][0], lm[0][3][1], lm[0][4][0], lm[0][4][1]]

            input = {
                'image': image,
                'landmarks': landmarks,
                'bounding_boxes': bounding_boxes,
                'index': index
            }
            X, bbx = self.transform(input)
            # TODO: Report -> This serves a check to see if each image is augmented differently in each epoch
            # if x == '003529.jpg' or x=='003530.jpg':
            #     import time
            #     ms = int(round(time.time() * 1000))
            #     save_image(X, self.config.basic.result_directory, x+str(ms))
        else:
            image = Image.open('{}/{}'.format(self.config.preprocessing.dataset.dataset_image_folder, x))
            X = self.transform(image)
            bbx = None

        # Save every X picture to validate preprocessing
        if self.config.preprocessing.save_preprocessed_image.enabled:
            if index % self.config.preprocessing.save_preprocessed_image.frequency == 0:
                save_input_transform_output_image(index, image, X, self.config.basic.result_directory, bbx)


        return X, y, index

    def get_attribute_baseline_accuracy(self):
        return self.labels.apply(pd.Series.value_counts).max() / len(self.labels)

    def get_attribute_baseline_majority_value(self):
        return self.labels.apply(pd.Series.value_counts).idxmax()

    def get_attribute_baseline_accuracy_val(self, train_attribute_baseline_majority_value):
        x = self.labels.apply(pd.Series.value_counts)
        lst = train_attribute_baseline_majority_value.tolist()
        access_tuple_list = [(0 if lst[y] == -1 else 1, y) for y in range(0, len(lst))]
        result_list = []
        for t in access_tuple_list:
            result_list.append((train_attribute_baseline_majority_value.keys()[t[1]], x.iloc[t] / self.labels.shape[0]))
        return pd.DataFrame(result_list).set_index(0)[1]

    def get_label_names(self):
        return self.labels.columns