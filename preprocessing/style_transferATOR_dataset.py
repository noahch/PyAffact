import torch
import pandas as pd
import numpy as np
from os import listdir
from os.path import isfile, join

from PIL import Image


class StyleTransferATORDataset(torch.utils.data.Dataset):
    def __init__(self, transform=None, max_size=None, index_offset=0, config=None, path=None):
        'Initialization'

        self.transform = transform
        if config:
            self.config = config
            self.dataset_path = 'dataset/' + self.config.preprocessing.dataset.path
        if path:
            self.dataset_path = 'dataset/' + path
        self.file_names = self._get_data()
        if max_size:
            self.data = self.file_names[index_offset:index_offset+max_size]

    def _get_data(self):
        return [f for f in listdir(self.dataset_path) if isfile(join(self.dataset_path, f))]

    def __len__(self):
        'Denotes the total number of samples'
        return len(self.data)


    def __getitem__(self, index):
        'Generates one sample of data'

        # Load data and get label
        X = Image.open(self.dataset_path + '/' + self.file_names[index])
        # X = np.array(X)
        if self.transform:
            X = self.transform(X)


        return X