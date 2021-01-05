import math

import matplotlib
import matplotlib.pyplot as plt
import torch
from PIL import Image
import numpy as np
import plotly.graph_objects as go

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

def save_image(output_image, result_directory, name):
    matplotlib.use('Agg')


    if isinstance(output_image, np.ndarray):
        output_image = Image.fromarray(np.transpose(output_image, (1, 2, 0)).astype(np.uint8), 'RGB')

    if isinstance(output_image, torch.Tensor):
        output_image = np.transpose(output_image.numpy(), (1, 2, 0))
        output_image = (output_image * 1 + 0) * 255
        output_image = output_image.astype(np.uint8)
        output_image = Image.fromarray(output_image, 'RGB')


    plt.clf()
    plt.figure(1)
    plt.imshow(output_image)
    plt.savefig('{}/{}.jpg'.format(result_directory, name))

def tensor_to_image(tensor):
    if isinstance(tensor, np.ndarray):
        output_image = Image.fromarray(np.transpose(tensor, (1, 2, 0)).astype(np.uint8), 'RGB')

    if isinstance(tensor, torch.Tensor):
        output_image = np.transpose(tensor.numpy(), (1, 2, 0))
        output_image = (output_image * 1 + 0) * 255
        output_image = output_image.astype(np.uint8)
        output_image = Image.fromarray(output_image, 'RGB')

    return output_image

def image_grid_and_accuracy_plot(images, accuracy_list, number_of_img_per_row=3, saveOnly = True):
    if saveOnly:
        matplotlib.use('Agg')
    else:
        matplotlib.use('TkAgg')
    plt.ion()
    plt.clf()
    plot_rows = math.ceil(len(images) / number_of_img_per_row)
    fig = plt.figure()
    fig.set_size_inches(5 * number_of_img_per_row, 5.5 * plot_rows)
    fig.tight_layout()
    for i in range(plot_rows):
        for j in range(number_of_img_per_row):
            current_index = ((i * number_of_img_per_row) + j)
            if current_index >= len(accuracy_list):
                break
            ax = plt.subplot2grid((plot_rows, number_of_img_per_row), (i, j))
            ax.set_title('Image-{}\nAccuracy: {}%'.format(current_index +1, accuracy_list[current_index]*100), fontsize=20)
            ax.axis('off')
            ax.imshow(tensor_to_image(images[current_index]))
    if not saveOnly:
        plt.show(block=True)

    return fig




    # fig.show()
    # plt.table(cellText=cellText, rowLabels=rows, colLabels=col)
    # plt.show(block=True)

