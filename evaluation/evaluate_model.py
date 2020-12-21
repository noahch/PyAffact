
import os

import bob
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import torch
import wandb
from torchvision.transforms.functional import to_tensor
from tqdm import tqdm

from evaluation.charts import generate_attribute_accuracy_chart, accuracy_table, generate_model_accuracy_of_testsets
from evaluation.utils import tensor_to_image, image_grid_and_accuracy_plot
from preprocessing.dataset_generator import generate_test_dataset
from training.model_manager import ModelManager


class EvalModel(ModelManager):
    def __init__(self, config, device):
        super().__init__(config, device)
        # load pickle based on config
        self.labels, _, _ = generate_test_dataset(config)






    def eval(self):
        checkpoint = torch.load('{}/{}'.format(self.config.experiments_dir, self.config.weights_name), map_location=self.config.basic.cuda_device_name.split(',')[0])
        self.model_device.load_state_dict(checkpoint['model_state_dict'])

        if self.config.evaluation.quantitative.enabled:
            self.quantitative_analysis(self.model_device)

        if self.config.evaluation.qualitative.enabled:
            self.qualitative_analysis(self.model_device)

    def quantitative_analysis(self, model):
        model.eval()

        per_attribute_accuracy_list = []
        all_attribute_accuracy_list = []

        # loop through all files of folder
        test_folders = os.listdir(self.config.dataset.testsets_path)

        for testset in test_folders:

            test_folder = os.path.join(self.config.dataset.testsets_path, testset)
            image_names = [f for f in os.listdir(test_folder) if os.path.isfile(os.path.join(test_folder, f))]
            pbar = tqdm(range(len(image_names)))
            correct_classifications = 0
            attributes_correct_count = torch.zeros(self.labels.shape[1])
            attributes_correct_count = attributes_correct_count.to(self.device)
            is_ten_crop = False

            for img in image_names:

                # Load image from disk yields numpy array of shape C x H x W
                img_path = '{}/{}/{}'.format(self.config.dataset.testsets_path, testset, img)
                # print(img_path)
                image = bob.io.base.load(img_path)

                if img[1] == '_':
                    is_ten_crop = True
                    img = img[2:]
                # Reshape to a numpy array of shape H x W x C
                image = np.transpose(image, (1, 2, 0))

                # convert each pixel to uint8
                image = image.astype(np.uint8)

                # to_tensor normalizes the numpy array (HxWxC) in the range [0. 255] to a torch.FloatTensor of shape (C x H x W) in the range [0.0, 1.0]
                image = to_tensor(image)


                # Get true Y label for image
                y = np.array(self.labels.loc[img].array)

                # -1 --> 0
                y = np.where(y < 0, 0, y)

                # image of shape H x W x C --> 1 x H x W x C
                inputs = image.unsqueeze(0)

                # Convert labels to tensor
                labels = torch.Tensor(y)

                # Also expand labels dimension by 1
                labels = labels.unsqueeze(0)

                # X on GPU
                inputs = inputs.to(self.device)

                # Y on GPU
                labels = labels.to(self.device)

                # track history if only in train
                with torch.no_grad():
                    outputs = self.model(inputs)
                    outputs[outputs < 0.5] = 0
                    outputs[outputs >= 0.5] = 1

                    attributes_correct_count += torch.sum(outputs == labels.type_as(outputs), dim=0)
                    correct_classifications += torch.sum(outputs == labels.type_as(outputs))
                pbar.update()

            per_attribute_accuracy = attributes_correct_count / len(image_names)
            per_attribute_accuracy_list.append(per_attribute_accuracy.tolist())
            all_attributes_accuracy = correct_classifications / (len(image_names) * self.labels.shape[1])
            all_attribute_accuracy_list.append(all_attributes_accuracy.tolist())

            pbar.close()

        #pd.DataFrame(per_attribute_accuracy.tolist(), columns=[testset])
        df = pd.DataFrame(np.transpose(per_attribute_accuracy_list), columns=test_folders)
        return df



    def qualitative_analysis(self, model):
        model.eval()
        data_iterator = iter(self.test_generator_A)
        images, labels, _ = data_iterator.next()
        inputs = images.to(self.device)
        labels = labels.to(self.device)

        # track history if only in train
        with torch.no_grad():
            outputs = self.model(inputs)
            outputs[outputs < 0.5] = 0
            outputs[outputs >= 0.5] = 1

        images = images.cpu()

        per_attribute_correct_classification = outputs == labels.type_as(outputs)
        accuracy_list = [round(x, 4) for x in
                         (torch.sum(per_attribute_correct_classification, dim=1) / self.labels.shape[1]).cpu().tolist()]
        prediction = outputs.type(torch.IntTensor).cpu().tolist()
        labels = labels.cpu().tolist()
        # print(prediction)
        # print(labels)
        # print(accuracy_list)

        per_attribute_correct_classification = per_attribute_correct_classification.cpu().tolist()

        accuracy_table_fig = accuracy_table(self.labels.columns.tolist(), prediction, per_attribute_correct_classification)
        accuracy_sample = image_grid_and_accuracy_plot(images, accuracy_list, number_of_img_per_row=self.config.evaluation.qualitative.number_of_images_per_row,
                                     result_directory=self.config.basic.result_directory, saveOnly=True)

        accuracy_sample.savefig('{}/accuracy_sample.jpg'.format(self.config.basic.result_directory))
        if self.config.basic.enable_wand_reporting:
            wandb.log({'Accuracy Sample Eval': accuracy_sample})
            wandb.log({'Accuracy Table Eval': accuracy_table_fig})
        accuracy_table_fig.write_image(os.path.join(self.config.basic.result_directory, 'acurracy_table.jpg'), scale=5)
        # plt.imshow(tensor_to_image(images[3]))
        # plt.show()
        # imshow(torchvision.utils.make_grid(images))


def imshow(img):
    image = tensor_to_image(img)
    plt.imshow(image)
    plt.show()
