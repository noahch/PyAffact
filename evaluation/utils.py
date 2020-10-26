import matplotlib
import matplotlib.pyplot as plt
import torch
from PIL import Image
import numpy as np

def save_input_transform_output_image(index, input_image, output_image, result_directory, bounding_box = None, saveOnly = True):
    if saveOnly:
        matplotlib.use('Agg')
    else:
        matplotlib.use('TkAgg')

    if isinstance(input_image, np.ndarray):
        input_image = Image.fromarray(np.transpose(input_image, (1, 2, 0)), 'RGB')

    if isinstance(output_image, np.ndarray):
        output_image = Image.fromarray(np.transpose(output_image, (1, 2, 0)).astype(np.uint8), 'RGB')

    if isinstance(output_image, torch.Tensor):
        output_image = np.transpose(output_image.numpy(), (1, 2, 0))
        output_image = (output_image * 1 + 0) * 255
        output_image = output_image.astype(np.uint8)
        output_image = Image.fromarray(output_image, 'RGB')

    plt.ion()
    plt.clf()
    plt.figure(1)
    plt.subplot(121)
    plt.imshow(input_image)
    if bounding_box:
        plt.plot([bounding_box[0], bounding_box[0], bounding_box[2], bounding_box[2]],
                 [bounding_box[1], bounding_box[3], bounding_box[1], bounding_box[3]],
                'rx', ms=10, mew=3)
    plt.subplot(122)
    plt.imshow(output_image)
    plt.savefig('{}/{}.jpg'.format(result_directory, index))