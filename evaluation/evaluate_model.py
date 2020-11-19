import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from torchvision.transforms import transforms

from evaluation.charts import generate_attribute_accuracy_chart
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
        self.landmarks = pd.read_pickle(
            os.path.join(self.config.basic.result_directory, self.config.evaluation.test_landmarks_pickle_filename),
            compression='zip')
        data_transforms = transforms.Compose([AffactTransformer(config)])
        self.dataset_test, self.test_generator = generate_dataset_and_loader(data_transforms, self.labels,
                                                                             self.landmarks, config)
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
        self.visualize_model(self.model_device)

    def visualize_model(self, model):
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
        figure = generate_attribute_accuracy_chart(self.labels.columns.tolist(), per_attribute_accuracy.tolist(), per_attribute_baseline_accuracy.tolist(), all_attributes_accuracy.tolist(), all_attributes_baseline_accuracy)
        figure.write_image(os.path.join(self.config.basic.result_directory, 'acurracy_plot.jpg'), scale=5)

