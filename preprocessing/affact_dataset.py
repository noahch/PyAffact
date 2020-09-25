import torch
import pandas as pd
import numpy as np
from skimage import io, transform
from PIL import Image

class AffactDataset(torch.utils.data.Dataset):
    def __init__(self, transform=None, max_size=None, index_offset=0):
        'Initialization'
        self.labels = pd.read_csv('dataset/CelebA/list_attr_celeba.txt', delim_whitespace=True)
        if max_size:
            self.labels = self.labels.iloc[index_offset:index_offset+max_size]
        self.transform = transform


    def __len__(self):
        'Denotes the total number of samples'
        return self.labels.shape[0]


    def __getitem__(self, index):
        'Generates one sample of data'
        # Select sample
        # ID = self.list_IDs[index]

        x = self.labels.iloc[index].name
        y = np.array(self.labels.iloc[index].array)
        y = np.where(y<0, 0, y)

        # Load data and get label
        # TODO: use pre_aligned flag from config
        X = Image.open('dataset/CelebA/img_align_celeba/' + x)

        if self.transform:
            X = self.transform(X)

        return X, y