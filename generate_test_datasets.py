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
    Starts creation of test datasets

    Returns
    -------

    """
    config = get_config('dataset/testsetAM_config')
    # Load dataframes for testing
    df_test_labels, df_test_landmarks, df_test_bounding_boxes = generate_test_dataset(config)
    # Define transformations for dataset AM (aligned, manual)
    # Consists of images aligned according to the hand-labled landmarks
    print('Creating testset AM (aligned, manual)')
    data_transforms_AM = transforms.Compose([AffactTransformer(config)])
    _create_test_images(config, df_test_labels, df_test_landmarks, df_test_bounding_boxes, data_transforms_AM)
    # # Define transformations for dataset AA (aligned, automatically)
    # # Consists of images aligned according to the face detector
    # print('Creating testset AA (aligned, automatically)')
    # config = get_config('dataset/testsetAA_config')
    # data_transforms_AA = transforms.Compose([AffactTransformer(config)])
    # _create_test_images(config, df_test_labels, df_test_landmarks, df_test_bounding_boxes, data_transforms_AA)
    # # face detector -> grösser bbx -> 10crop
    # config = get_config('dataset/testsetC_config')
    # print('Creating testset C (10 crop)')
    # data_transforms_C = transforms.Compose([AffactTransformer(config), TenCrop(224)])
    # _create_test_images(config, df_test_labels, df_test_landmarks, df_test_bounding_boxes, data_transforms_C,
    #                    uses_tencrop=True)
    # # face detector -> bigger bbx -> AFFACT Transformations
    # config = get_config('dataset/testsetT_config')
    # print('Creating testset T (AFFACT transformations)')
    # data_transforms_T = transforms.Compose([AffactTransformer(config)])
    # _create_test_images(config, df_test_labels, df_test_landmarks, df_test_bounding_boxes, data_transforms_T)



def _create_test_images(config, df_test_labels, df_test_landmarks, df_test_bounding_boxes, transformer, uses_tencrop=False):
    """
    Generates test images based on dataframes

    Parameters
    ----------
    config Configuration File
    df_test_labels labels dataframe
    df_test_landmarks Landmark dataframe
    df_test_bounding_boxes bounding boxes dataframe
    transformer transformer

    Returns
    -------

    """

    # TODO
    # create_directory(config.dataset_result_folder, recreate=True)
    create_directory(config.dataset_result_folder)
    print("created {}".format(config.dataset_result_folder))
    pbar = tqdm(range(len(df_test_labels.index)))
    # pbar.clear()
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
            scale = 2.5

            # If the MTCNN cannot find a bounding box, we load the bounding box from the disk
            try:
                bounding_boxes = bounding_boxes[0]
                bounding_boxes[2] = bounding_boxes[2] - bounding_boxes[0]
                bounding_boxes[3] = bounding_boxes[3] - bounding_boxes[1]
            except:
                print(row.name)
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

        if not uses_tencrop:
            img = tensor_to_image(X)
            img.save('{}/{}'.format(config.dataset_result_folder, row.name))
        else:
            for i in range(10):
                img = tensor_to_image(X[i])
                img.save('{}/{}_{}'.format(config.dataset_result_folder, i, row.name))

        pbar.update(1)
    pbar.close()


if __name__ == '__main__':
    main()

