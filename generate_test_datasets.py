import bob
import pandas as pd
from PIL import Image
from facenet_pytorch.models.mtcnn import MTCNN

import numpy as np
from torchvision.transforms import transforms, TenCrop
from tqdm import tqdm

from preprocessing.dataset_generator import generate_test_dataset
from utils.config_utils import get_config
from evaluation.utils import tensor_to_image
from preprocessing.affact_transformer import AffactTransformer
from utils.utils import create_directory



def create_test_images(config, df_test_labels, df_test_landmarks, df_test_bounding_boxes, transformer):
    """

    Parameters
    ----------
    config
    df_test_landmarks
    df_test_bounding_boxes
    transformer

    Returns
    -------

    """

    create_directory(config.dataset_result_folder, recreate=True)
    pbar = tqdm(range(len(df_test_labels.index)))
    pbar.clear()

    for index, (i, row) in enumerate(df_test_labels.iterrows()):
        image = bob.io.base.load('{}/{}'.format(config.preprocessing.dataset.dataset_image_folder, row.name))
        landmarks, bounding_boxes = None, None
        if config.preprocessing.dataset.bounding_box_mode == 0:
            landmarks = df_test_landmarks.iloc[index].tolist()
            landmarks = landmarks[:4] + landmarks[6:]
        elif config.preprocessing.dataset.bounding_box_mode == 1:
            bounding_boxes = df_test_bounding_boxes.iloc[index].tolist()
            bounding_boxes = bounding_boxes[1:]
        elif config.preprocessing.dataset.bounding_box_mode == 2:
            mtcnn = MTCNN(select_largest=False, device=config.basic.cuda_device_name.split(',')[0])
            boxes, probs, lm = mtcnn.detect(Image.fromarray(np.transpose(image, (1, 2, 0)), 'RGB'), landmarks=True)
            landmarks = [lm[0][0][0], lm[0][0][1], lm[0][1][0], lm[0][1][1],
                         lm[0][3][0], lm[0][3][1], lm[0][4][0], lm[0][4][1]]

        input = {
            'image': image,
            'landmarks': landmarks,
            'bounding_boxes': bounding_boxes,
            'index': index
        }
        X = transformer(input)

        if config.preprocessing.dataset.bounding_box_mode != 2:
            img = tensor_to_image(X)
            img.save('{}/{}'.format(config.dataset_result_folder, row.name))
        else:

        pbar.update(1)







config = get_config('testsetAM_config')

# Load dataframes for testing
df_test_labels, df_test_landmarks, df_test_bounding_boxes = generate_test_dataset(config)


# Define transformations for dataset AM (aligned, manual)
# Consists of images aligned according to the hand-labled landmarks
data_transforms_A = transforms.Compose([AffactTransformer(config), TenCrop(224)])

# print('Creating testset AM (aligned, manual)')
# create_test_images(config, df_test_labels, df_test_landmarks, df_test_bounding_boxes, data_transforms_A)

# face detector -> grösser bbx -> 10crop

image = bob.io.base.load('{}/{}'.format(config.preprocessing.dataset.dataset_image_folder, '182638.jpg'))
landmarks, bounding_boxes = None, None

mtcnn = MTCNN(margin=40, select_largest=False, post_process=False, device='cuda:0')
bounding_boxes, probs, lm = mtcnn.detect(Image.fromarray(np.transpose(image, (1, 2, 0)), 'RGB'), landmarks=True)
# print(bounding_boxes)
scale = 2.5
bounding_boxes = bounding_boxes[0]

bounding_boxes[2] = bounding_boxes[2] - bounding_boxes[0]
bounding_boxes[3] = bounding_boxes[3] - bounding_boxes[1]


bounding_boxes[0] = bounding_boxes[0] - ((scale-1)/2 * bounding_boxes[2])
bounding_boxes[1] = bounding_boxes[1] - ((scale-1)/2 * bounding_boxes[3])
bounding_boxes[2] = scale * (bounding_boxes[2])
bounding_boxes[3] = scale * (bounding_boxes[3])
print(bounding_boxes)


input = {
    'image': image,
    'landmarks': landmarks,
    'bounding_boxes': bounding_boxes,
    'index': 0
}
X = data_transforms_A(input)

img = tensor_to_image(X[6])
img.save('image2.jpg')


# face detector -> bounding box -> transformation






# read data and generate folder with transformed images
