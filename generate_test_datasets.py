import random

import bob
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


def main():
    """
    Creates the test datasets needed for evaluation
    """

    config = get_config('dataset/testsetA_config')
    # Load dataframes for testing
    df_test_labels, df_test_landmarks, df_test_bounding_boxes = generate_test_dataset(config)
    # Define transformations for dataset A (aligned)
    # Consists of images aligned according to the hand-labled landmarks
    print('Creating testset A (aligned)')
    data_transforms_A = transforms.Compose([AffactTransformer(config)])
    _create_test_images(config, df_test_labels, df_test_landmarks, df_test_bounding_boxes, data_transforms_A)

    # Define transformations for dataset D (detected bounding boxes)
    # Consists of images aligned according to the face detector
    print('Creating testset D (detected bounding boxes)')
    config = get_config('dataset/testsetD_config')
    data_transforms_D= transforms.Compose([AffactTransformer(config)])
    _create_test_images(config, df_test_labels, df_test_landmarks, df_test_bounding_boxes, data_transforms_D)

    # face detector -> grösser bbx -> 10crop
    config = get_config('dataset/testsetC_config')
    print('Creating testset C (10 crop)')
    data_transforms_C = transforms.Compose([AffactTransformer(config)])
    _create_test_images(config, df_test_labels, df_test_landmarks, df_test_bounding_boxes, data_transforms_C)

    # face detector -> bigger bbx -> AFFACT Transformations
    config = get_config('dataset/testsetT_config')
    print('Creating testset T (AFFACT transformations)')
    data_transforms_T = transforms.Compose([AffactTransformer(config)])
    _create_test_images(config, df_test_labels, df_test_landmarks, df_test_bounding_boxes, data_transforms_T)




def _create_test_images(config, df_test_labels, df_test_landmarks, df_test_bounding_boxes, transformer):
    """
    Generates test images based on dataframes
    :param config: Configuration File
    :param df_test_labels: labels dataframe
    :param df_test_landmarks: Landmark dataframe
    :param df_test_bounding_boxes: bounding boxes dataframe
    :param transformer: transformer
    """


    create_directory(config.dataset_result_folder, recreate=True)
    print("created {}".format(config.dataset_result_folder))
    pbar = tqdm(range(len(df_test_labels.index)))

    mtcnn = MTCNN(select_largest=False, post_process=False, device='cuda:0')

    for index, (i, row) in enumerate(df_test_labels.iterrows()):
        image = bob.io.base.load('{}/{}'.format(config.dataset.dataset_image_folder, row.name))
        landmarks, bounding_boxes = None, None
        if config.dataset.bounding_box_mode == 0:
            landmarks = df_test_landmarks.iloc[index].tolist()
            landmarks = landmarks[:4] + landmarks[6:]
        elif config.dataset.bounding_box_mode == 1:
            bounding_boxes = df_test_bounding_boxes.iloc[index].tolist()
            bounding_boxes = bounding_boxes[1:]
        elif config.dataset.bounding_box_mode == 2:
            bounding_boxes, probs, lm = mtcnn.detect(Image.fromarray(np.transpose(image, (1, 2, 0)), 'RGB'),
                                                     landmarks=True)
            # print(bounding_boxes)
            scale = config.dataset.bounding_box_scale

            # If the MTCNN cannot find a bounding box, we load the bounding box from the disk
            try:
                bounding_boxes = bounding_boxes[0]
                bounding_boxes[2] = bounding_boxes[2] - bounding_boxes[0]
                bounding_boxes[3] = bounding_boxes[3] - bounding_boxes[1]
            except:
                # print(row.name)
                bounding_boxes = df_test_bounding_boxes.iloc[index].tolist()
                bounding_boxes = bounding_boxes[1:]


            bounding_boxes[0] = bounding_boxes[0] - ((scale - 1) / 2 * bounding_boxes[2])
            bounding_boxes[1] = bounding_boxes[1] - ((scale - 1) / 2 * bounding_boxes[3])
            bounding_boxes[2] = scale * (bounding_boxes[2])
            bounding_boxes[3] = scale * (bounding_boxes[3])

        input = {
            'image': image,
            'landmarks': landmarks,
            'bounding_boxes': bounding_boxes,
            'index': index
        }
        X = transformer(input)


        img = tensor_to_image(X)
        img.save('{}/{}'.format(config.dataset_result_folder, row.name[:-3] + 'png'))


        pbar.update(1)
    pbar.close()


if __name__ == '__main__':
    random.seed(a=0, version=2)
    np.random.seed(0)
    main()

