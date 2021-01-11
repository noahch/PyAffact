import matplotlib
import matplotlib.pyplot as plt
import torch
from PIL import Image
import numpy as np


def save_original_and_preprocessed_image(filename, original_image, preprocessed_image, result_directory):
    """
    saves the original and preprocessed image to disk
    :param filename: the name of the file
    :param original_image: original image
    :param preprocessed_image: preprocessed torch/numpy image ready for model
    :param result_directory: path to the result directory
    """

    matplotlib.use('Agg')

    if isinstance(original_image, np.ndarray):
        original_image = Image.fromarray(np.transpose(original_image, (1, 2, 0)), 'RGB')

    if isinstance(preprocessed_image, np.ndarray):
        preprocessed_image = Image.fromarray(np.transpose(preprocessed_image, (1, 2, 0)).astype(np.uint8), 'RGB')

    if isinstance(preprocessed_image, torch.Tensor):
        preprocessed_image = np.transpose(preprocessed_image.numpy(), (1, 2, 0))
        preprocessed_image = (preprocessed_image * 1 + 0) * 255
        preprocessed_image = preprocessed_image.astype(np.uint8)
        preprocessed_image = Image.fromarray(preprocessed_image, 'RGB')

    plt.ion()
    plt.clf()
    plt.figure(1)
    plt.subplot(121)
    plt.imshow(original_image)
    plt.subplot(122)
    plt.imshow(preprocessed_image)
    plt.savefig('{}/{}.jpg'.format(result_directory, filename))


def tensor_to_image(tensor):
    """
    helper function to convert a tensor to a PIL image
    :param tensor: numpy ndarray/torch tensor
    :return: PIL image
    """
    output_image = None
    if isinstance(tensor, np.ndarray):
        output_image = Image.fromarray(np.transpose(tensor, (1, 2, 0)).astype(np.uint8), 'RGB')

    elif isinstance(tensor, torch.Tensor):
        output_image = np.transpose(tensor.numpy(), (1, 2, 0))
        output_image = (output_image * 1 + 0) * 255
        output_image = output_image.astype(np.uint8)
        output_image = Image.fromarray(output_image, 'RGB')

    return output_image
