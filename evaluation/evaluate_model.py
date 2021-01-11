
import os

import bob
import numpy as np
import pandas as pd
import torch
from torchvision.transforms import transforms, TenCrop, ToTensor
from torchvision.transforms.functional import to_tensor, normalize
from tqdm import tqdm

from evaluation.charts import prediction_vs_ground_truth_chart, generate_model_accuracy_of_testsets, \
    accuracy_sample_plot
from evaluation.utils import image_grid_and_accuracy_plot
from preprocessing.dataset_generator import generate_test_dataset
from training.model_manager import ModelManager


class EvalModel(ModelManager):
    def __init__(self, config, device):
        super().__init__(config, device)
        # load pickle based on config
        self.labels, _, _ = generate_test_dataset(config)

    def evaluate(self):
        checkpoint = torch.load('{}/{}'.format(self.config.experiments_dir, self.config.weights_name),
                                map_location=self.config.basic.cuda_device_name.split(',')[0])
        self.model_device.load_state_dict(checkpoint['model_state_dict'])
        self.model_device.eval()

        per_attribute_accuracy_list = []

        # loop through all files of folder
        test_folders = os.listdir(self.config.dataset.testsets_path)
        test_folders.sort()
        test_folders.remove('testsetC')

        for testset in test_folders:
            print("Calculating Accuracy for: {}".format(testset))
            test_folder = os.path.join(self.config.dataset.testsets_path, testset)
            image_names = [f for f in os.listdir(test_folder) if os.path.isfile(os.path.join(test_folder, f))]
            pbar = tqdm(range(len(image_names)))
            attributes_correct_count = torch.zeros(self.labels.shape[1])
            attributes_correct_count = attributes_correct_count.to(self.device)

            batch_size = self.config.evaluation.batch_size

            image_name_batches = [image_names[i:i + batch_size] for i in range(0, len(image_names), batch_size)]

            for img_batch in image_name_batches:
                inputs = []
                labels = []
                for img in img_batch:
                    # load image and label as tensors
                    image, y, _ = self._image_and_label_to_tensor(img, testset)

                    # Add image to inputs batch list
                    inputs.append(image)

                    # Add label to labels batch list
                    labels.append(y)

                # turn list of tensors to 1 tensor of shape batch size C x H x W
                inputs = torch.stack(inputs)

                # turn list of tensors to 1 tensor of shape batch size x number of labels
                labels = torch.stack(labels)

                # inputs on GPU
                inputs = inputs.to(self.device)

                # labels on GPU
                labels = labels.to(self.device)

                # do not track history if inference
                with torch.no_grad():
                    outputs = self.model_device(inputs)
                    outputs[outputs < 0.5] = 0
                    outputs[outputs >= 0.5] = 1

                    attributes_correct_count += torch.sum(outputs == labels.type_as(outputs), dim=0)
                pbar.update(len(img_batch))

            per_attribute_accuracy = attributes_correct_count / len(image_names)
            per_attribute_accuracy_list.append(per_attribute_accuracy.tolist())
            pbar.close()


        # validate 10 crop
        testset = 'testsetC'
        print("Calculating Accuracy for: {}".format(testset))
        test_folder = os.path.join(self.config.dataset.testsets_path, testset)
        image_names = [f for f in os.listdir(test_folder) if os.path.isfile(os.path.join(test_folder, f))]
        pbar = tqdm(range(len(image_names)))
        attributes_correct_count = torch.zeros(self.labels.shape[1])
        attributes_correct_count = attributes_correct_count.to(self.device)
        for img in image_names:
            image, y, _ = self._image_and_label_to_tensor(img, testset)
            transf = transforms.Compose([TenCrop(224)])
            x = transf(image)

            # turn list of tensors to 1 tensor of shape batch size C x H x W
            inputs = torch.stack(x)

            # turn list of tensors to 1 tensor of shape batch size x number of labels
            labels = torch.stack([y]*10)

            # inputs on GPU
            inputs = inputs.to(self.device)

            # labels on GPU
            labels = labels.to(self.device)

            # do not track history if inference
            with torch.no_grad():
                outputs = self.model_device(inputs)

                # round percentages to 0 or 1
                outputs[outputs < 0.5] = 0
                outputs[outputs >= 0.5] = 1

                # sum the outputs of the 10 crops
                outputs_sum = torch.sum(outputs, dim=0)  # size = [1, ncol]

                # take majority guess
                outputs_sum[outputs_sum < 5] = 0
                outputs_sum[outputs_sum >= 5] = 1

                attributes_correct_count += torch.sum(outputs_sum == labels[0].unsqueeze(0), dim=0)
            pbar.update(1)

        per_attribute_accuracy = attributes_correct_count / len(image_names)
        per_attribute_accuracy_list.append(per_attribute_accuracy.tolist())

        pbar.close()
        test_folders.append('testsetC')

        df = pd.DataFrame(np.transpose(per_attribute_accuracy_list), columns=test_folders)
        return df

    def quantitative_analysis(self):
        # Load labels
        labels, _, _ = generate_test_dataset(self.config)

        # Load accuracy DF from disk
        accuracy_df = pd.read_csv('{}/evaluation_result.csv'.format(self.config.experiments_dir),
                                  index_col=0)

        # Calculate baseline Accuracies
        test_attribute_baseline_accuracy = self._get_attribute_baseline_accuracy_val(labels, self.config)
        all_attributes_baseline_accuracy = test_attribute_baseline_accuracy.sum(axis=0) / labels.shape[1]
        per_attribute_baseline_accuracy = test_attribute_baseline_accuracy

        # Generate figure with accuracies on different test sets for model and baseline
        figure = generate_model_accuracy_of_testsets(labels.columns.tolist(), accuracy_df,
                                                     per_attribute_baseline_accuracy.tolist(),
                                                     all_attributes_baseline_accuracy)
        # Save the figure
        figure.write_image('{}/eval_fig.png'.format(self.config.experiments_dir), format='png', scale=3)
        figure.show()

    def qualitative_analysis(self):
        checkpoint = torch.load('{}/{}'.format(self.config.experiments_dir, self.config.weights_name),
                                map_location=self.config.basic.cuda_device_name.split(',')[0])
        self.model_device.load_state_dict(checkpoint['model_state_dict'])
        self.model_device.eval()

        test_folder = os.path.join(self.config.dataset.testsets_path, self.config.evaluation.qualitative.testset_name)
        image_names = [f for f in os.listdir(test_folder) if os.path.isfile(os.path.join(test_folder, f))]

        inputs = []
        labels = []
        original_images = []
        for i in range(self.config.evaluation.qualitative.number_of_images_per_row * self.config.evaluation.qualitative.number_of_rows):
            img = image_names[i]

            # load image and label as tensors
            image, y, original_img = self._image_and_label_to_tensor(img, self.config.evaluation.qualitative.testset_name)

            # Add image to inputs batch list
            inputs.append(image)

            original_images.append(original_img)

            # Add label to labels batch list
            labels.append(y)

        # turn list of tensors to 1 tensor of shape batch size C x H x W
        inputs = torch.stack(inputs)

        # turn list of tensors to 1 tensor of shape batch size x number of labels
        labels = torch.stack(labels)

        # inputs on GPU
        inputs = inputs.to(self.device)

        # labels on GPU
        labels = labels.to(self.device)

        # do not track history if inference
        with torch.no_grad():
            outputs = self.model_device(inputs)
            outputs[outputs < 0.5] = 0
            outputs[outputs >= 0.5] = 1

        images = inputs.cpu()

        per_attribute_correct_classification = outputs == labels.type_as(outputs)
        accuracy_list = [round(x, 4) for x in
                         (torch.sum(per_attribute_correct_classification, dim=1) / self.labels.shape[1]).cpu().tolist()]
        prediction = outputs.type(torch.IntTensor).cpu().tolist()

        per_attribute_correct_classification = per_attribute_correct_classification.cpu().tolist()

        accuracy_table_fig = prediction_vs_ground_truth_chart(self.labels.columns.tolist(), prediction,
                                                              per_attribute_correct_classification)
        accuracy_sample = accuracy_sample_plot(original_images, accuracy_list,
                                                       number_of_img_per_row=self.config.evaluation.qualitative.number_of_images_per_row)

        accuracy_sample.savefig(os.path.join(self.config.experiments_dir, 'accuracy_sample.jpg'))

        accuracy_table_fig.write_image(os.path.join(self.config.experiments_dir, 'prediction_vs_ground_truth_table.jpg'), scale=5)

    def _image_and_label_to_tensor(self, img_name, testset):
        orig_image, img_name = self._load_image(img_name, testset)

        # Reshape to a numpy array of shape H x W x C
        image = np.transpose(orig_image, (1, 2, 0))

        # convert each pixel to uint8
        image = image.astype(np.uint8)

        # to_tensor normalizes the numpy array (HxWxC) in the range [0. 255] to a torch.FloatTensor of shape (C x H x W) in the range [0.0, 1.0]
        image = normalize(to_tensor(image), mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

        # Get true Y label for image
        y = np.array(self.labels.loc[img_name].array)

        # -1 --> 0
        y = np.where(y < 0, 0, y)

        return image, torch.Tensor(y), orig_image

    def _load_image(self, img_name, testset):
        # Load image from disk yields numpy array of shape C x H x W
        img_path = '{}/{}/{}'.format(self.config.dataset.testsets_path, testset, img_name)

        img_name = img_name[:-3] + 'jpg'
        # print(img_path)
        image = bob.io.base.load(img_path)
        return image, img_name

    def _get_attribute_baseline_accuracy_val(self, labels, config):
        train_attribute_baseline_majority_value = pd.read_pickle(config.dataset.majority_class_file,
                                                                 compression='zip')
        x = labels.apply(pd.Series.value_counts)
        lst = train_attribute_baseline_majority_value.tolist()
        access_tuple_list = [(0 if lst[y] == -1 else 1, y) for y in range(0, len(lst))]
        result_list = []
        for t in access_tuple_list:
            result_list.append((train_attribute_baseline_majority_value.keys()[t[1]], x.iloc[t] / labels.shape[0]))
        return pd.DataFrame(result_list).set_index(0)[1]