import os

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torchvision
import wandb
from torchvision.transforms import transforms

from evaluation.charts import generate_attribute_accuracy_chart, accuracy_table
from evaluation.utils import tensor_to_image, image_grid_and_accuracy_plot
from preprocessing.affact_transformer import AffactTransformer
from preprocessing.utils import get_train_val_dataset, generate_dataset_and_loader
from training.model_manager import ModelManager


class EvalModel(ModelManager):
    def __init__(self, config, device):
        super().__init__(config, device)
        # load pickle based on config
        self.labels = pd.read_pickle(
            os.path.join(self.config.basic.result_directory, self.config.evaluation.test_labels_pickle_filename),
            compression='zip')

        self.landmarks, self.bounding_boxes = None, None
        if self.config.preprocessing.dataset.uses_landmarks:
            self.landmarks = pd.read_pickle(
                os.path.join(self.config.basic.result_directory, self.config.evaluation.test_landmarks_pickle_filename),
                compression='zip')
        else:
            self.bounding_boxes = pd.read_pickle(
                os.path.join(self.config.basic.result_directory, self.config.evaluation.test_bounding_boxes_filename),
                compression='zip')

        # TODO: Switch off
        data_transforms = transforms.Compose([AffactTransformer(config)])
        # data_transforms = transforms.Compose([
        #     transforms.CenterCrop([224, 224]),
        #     # transforms.Resize([224, 224]),
        #     transforms.ToTensor()
        # ])
        self.dataset_test, self.test_generator = generate_dataset_and_loader(data_transforms, self.labels,
                                                                             self.landmarks, self.bounding_boxes, config)
        train_attribute_baseline_majority_value = pd.read_pickle(
            os.path.join(self.config.basic.result_directory, self.config.evaluation.train_majority_pickle_filename),
            compression='zip')
        self.test_attribute_baseline_accuracy = self.dataset_test.get_attribute_baseline_accuracy_val(
            train_attribute_baseline_majority_value)
        self.optimizer = self._get_optimizer()

    def eval(self):
        checkpoint = torch.load(
            os.path.join(self.config.basic.result_directory, self.config.evaluation.model_weights_filename))
        self.model_device.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

        if self.config.evaluation.quantitative.enabled:
            self.quantitative_analysis(self.model_device)

        if self.config.evaluation.qualitative.enabled:
            self.qualitative_analysis(self.model_device)

    def quantitative_analysis(self, model):
        model.eval()

        correct_classifications = 0
        attributes_correct_count = torch.zeros(self.labels.shape[1])
        attributes_correct_count = attributes_correct_count.to(self.device)
        for inputs, labels, _ in self.test_generator:
            inputs = inputs.to(self.device)
            labels = labels.to(self.device)

            # track history if only in train
            with torch.no_grad():
                outputs = self.model(inputs)
                outputs[outputs < 0.5] = 0
                outputs[outputs >= 0.5] = 1

                attributes_correct_count += torch.sum(outputs == labels.type_as(outputs), dim=0)
                correct_classifications += torch.sum(outputs == labels.type_as(outputs))

        per_attribute_accuracy = attributes_correct_count / self.labels.shape[0]
        per_attribute_baseline_accuracy = self.test_attribute_baseline_accuracy
        all_attributes_accuracy = correct_classifications / (self.labels.shape[0] * self.labels.shape[1])
        all_attributes_baseline_accuracy = self.test_attribute_baseline_accuracy.sum(axis=0) / self.labels.shape[1]
        figure = generate_attribute_accuracy_chart(self.labels.columns.tolist(), per_attribute_accuracy.tolist(),
                                                   per_attribute_baseline_accuracy.tolist(),
                                                   all_attributes_accuracy.tolist(), all_attributes_baseline_accuracy)
        if self.config.basic.enable_wand_reporting:
            wandb.log({'Accuracy Plot Eval': figure})
        figure.write_image(os.path.join(self.config.basic.result_directory, 'acurracy_plot.jpg'), scale=5)

    def qualitative_analysis(self, model):
        model.eval()
        matplotlib.use('TkAgg')
        data_iterator = iter(self.test_generator)
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
