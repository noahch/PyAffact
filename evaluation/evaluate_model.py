"""
class that evaluates a model
"""
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
from preprocessing.dataset_generator import generate_test_dataset
from training.model_manager import ModelManager


class EvalModel(ModelManager):
    """
    Class that evaluates a model
    """
    def __init__(self, config, device):
        """
        Initialization
        :param config: evaluation configuration file
        :param device: Cuda Device
        """
        super().__init__(config, device)
        # get the labels of the test data
        self.labels, _, _ = generate_test_dataset(config)

    def evaluate(self):
        """
        Calculate the accuracy of the model on different test sets
        :return: dataframe containing the attribute accuracies on different test sets
        """

        # Load the weights
        checkpoint = torch.load('{}/{}'.format(self.config.experiments_dir, self.config.weights_name),
                                map_location=self.config.basic.cuda_device_name.split(',')[0])

        # Initialize the model with the weigths
        self.model_device.load_state_dict(checkpoint['model_state_dict'])

        # Set the model to evaluation mode
        self.model_device.eval()

        per_attribute_accuracy_list = []

        # loop through all files of folder
        test_folders = os.listdir(self.config.dataset.testsets_path)
        # sort files
        test_folders.sort()

        # remove folder testsetC, as it has to be evaluated differently
        test_folders.remove('testsetC')

        # Loop through test sets
        for testset in test_folders:
            print("Calculating Accuracy for: {}".format(testset))

            # Get the folder containing the images
            test_folder = os.path.join(self.config.dataset.testsets_path, testset)

            # Get image names
            image_names = [f for f in os.listdir(test_folder) if os.path.isfile(os.path.join(test_folder, f))]

            # Initialize progress bar
            progress_bar = tqdm(range(len(image_names)))

            # Initialize structure to count correct classifications per attribute
            attributes_correct_count = torch.zeros(self.labels.shape[1])
            attributes_correct_count = attributes_correct_count.to(self.device)

            # load the batch size
            batch_size = self.config.evaluation.batch_size

            # create image batches
            image_name_batches = [image_names[i:i + batch_size] for i in range(0, len(image_names), batch_size)]

            # for each batch
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
                    # Calculate predictions
                    predictions = self.model_device(inputs)
                    # Remap predictions from probability to yes or no
                    predictions[predictions < 0.5] = 0
                    predictions[predictions >= 0.5] = 1

                    # Count correct classifications per attribute
                    attributes_correct_count += torch.sum(predictions == labels.type_as(predictions), dim=0)

                # Update progress bar
                progress_bar.update(len(img_batch))

            # Calculate per attribute accuracy
            per_attribute_accuracy = attributes_correct_count / len(image_names)

            # Save per attribute accuracy per test set
            per_attribute_accuracy_list.append(per_attribute_accuracy.tolist())

            # Close progress bar
            progress_bar.close()

        # Separate Validation for 10 Crop
        # Name of 10 crop test set
        testset = 'testsetC'
        print("Calculating Accuracy for: {}".format(testset))

        # Get the folder containing the images
        test_folder = os.path.join(self.config.dataset.testsets_path, testset)

        # Get the image names
        image_names = [f for f in os.listdir(test_folder) if os.path.isfile(os.path.join(test_folder, f))]

        # Initialize progress bar
        progress_bar = tqdm(range(len(image_names)))

        # Initialize structure to count correct classifications per attribute
        attributes_correct_count = torch.zeros(self.labels.shape[1])
        attributes_correct_count = attributes_correct_count.to(self.device)

        # For each image
        for img in image_names:
            # load image and label as tensors
            image, y, _ = self._image_and_label_to_tensor(img, testset)

            # Apply 10 Crop to the image
            transformer = transforms.Compose([TenCrop(224)])
            x = transformer(image)

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
                # Calculate predictions
                predictions = self.model_device(inputs)

                # Remap predictions from probability to yes or no
                predictions[predictions < 0.5] = 0
                predictions[predictions >= 0.5] = 1

                # sum the predictions of the 10 crops
                outputs_sum = torch.sum(predictions, dim=0)

                # take majority guess
                outputs_sum[outputs_sum < 5] = 0
                outputs_sum[outputs_sum >= 5] = 1

                # Count correct classifications per attribute
                attributes_correct_count += torch.sum(outputs_sum == labels[0].unsqueeze(0), dim=0)

            # Update progress bar
            progress_bar.update(1)

        # Calculate per attribute accuracy
        per_attribute_accuracy = attributes_correct_count / len(image_names)

        # Save per attribute accuracy per test set
        per_attribute_accuracy_list.append(per_attribute_accuracy.tolist())

        # Close progress bar
        progress_bar.close()

        # Add testsetC folder back to all folders
        test_folders.append('testsetC')

        # Create dataframe containing the accuracies
        df = pd.DataFrame(np.transpose(per_attribute_accuracy_list), columns=test_folders)
        return df

    def quantitative_analysis(self, order=None):
        """
        Loads the evaluation_result.csv file and quantitatively evaluates the model
        Saves the evaluation chart and displays it in the browser
        :param order: Display order of test sets
        """
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
                                                     all_attributes_baseline_accuracy, order=order)
        # Save the figure
        figure.write_image('{}/eval_fig.png'.format(self.config.experiments_dir), format='png', scale=3)
        figure.show()

    def qualitative_analysis(self):
        """
        Loads the model weights and qualitatively evaluates the model
        Saves the figure with samples as well as a ground truth table
        """

        # Load model weights
        checkpoint = torch.load('{}/{}'.format(self.config.experiments_dir, self.config.weights_name),
                                map_location=self.config.basic.cuda_device_name.split(',')[0])

        # Load weights into model
        self.model_device.load_state_dict(checkpoint['model_state_dict'])

        # Set model to evaluation mode
        self.model_device.eval()

        # Get the test set folder for evaluation
        test_folder = os.path.join(self.config.dataset.testsets_path, self.config.evaluation.qualitative.testset_name)

        # Get the image names
        image_names = [f for f in os.listdir(test_folder) if os.path.isfile(os.path.join(test_folder, f))]

        inputs = []
        labels = []
        original_images = []

        # for x amount of images, where x is number of images per row times number of rows
        for i in range(self.config.evaluation.qualitative.number_of_images_per_row * self.config.evaluation.qualitative.number_of_rows):

            # Current image name
            image_name = image_names[i]

            # load image and label as tensors
            image, y, original_img = self._image_and_label_to_tensor(image_name, self.config.evaluation.qualitative.testset_name)

            # Add image to inputs batch list
            inputs.append(image)

            # save original image without transformations
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
            # Calculate predictions
            predictions = self.model_device(inputs)

            # Remap predictions from probability to yes or no
            predictions[predictions < 0.5] = 0
            predictions[predictions >= 0.5] = 1

        # Calculate per attribute correct classifications
        per_attribute_correct_classification = predictions == labels.type_as(predictions)

        # Save accuracies to list
        accuracy_list = [round(x, 4) for x in
                         (torch.sum(per_attribute_correct_classification, dim=1) / self.labels.shape[1]).cpu().tolist()]

        # Save predictions to list
        prediction = predictions.type(torch.IntTensor).cpu().tolist()

        # Transfer and convert per attribute correct classification to list
        per_attribute_correct_classification = per_attribute_correct_classification.cpu().tolist()

        # Generate prediction vs ground truth chart
        accuracy_table_fig = prediction_vs_ground_truth_chart(self.labels.columns.tolist(), prediction,
                                                              per_attribute_correct_classification)
        # Save prediction vs ground truth chart
        accuracy_table_fig.write_image(
            os.path.join(self.config.experiments_dir, 'prediction_vs_ground_truth_table.jpg'), scale=5)

        # Generate grid with sample images
        accuracy_sample = accuracy_sample_plot(original_images, accuracy_list,
                                                       number_of_img_per_row=self.config.evaluation.qualitative.number_of_images_per_row)

        # Save grid with sample images
        accuracy_sample.savefig(os.path.join(self.config.experiments_dir, 'accuracy_sample.jpg'))



    def _image_and_label_to_tensor(self, img_name, testset):
        """
        Load and image and the respective label as a tensor
        :param img_name: name of the image
        :param testset: test set to load the image from
        :return: transformed image (Tensor), label (Tensor), original image (Image)
        """

        # Load image
        original_image, img_name = self._load_image(img_name, testset)

        # Reshape to a numpy array of shape H x W x C
        image = np.transpose(original_image, (1, 2, 0))

        # convert each pixel to uint8
        image = image.astype(np.uint8)

        # to_tensor normalizes the numpy array (HxWxC) in the range [0. 255] to a torch.FloatTensor of shape (C x H x W) in the range [0.0, 1.0]
        image = normalize(to_tensor(image), mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

        # Get true Y label for image
        y = np.array(self.labels.loc[img_name].array)

        # -1 --> 0
        y = np.where(y < 0, 0, y)

        return image, torch.Tensor(y), original_image

    def _load_image(self, img_name, testset):
        """
        Load image from disk
        :param img_name: name of the image
        :param testset: test set folder to load image from
        :return: image, image name
        """
        # Load image from disk yields numpy array of shape C x H x W
        img_path = '{}/{}/{}'.format(self.config.dataset.testsets_path, testset, img_name)

        # Change name back to JPG
        img_name = img_name[:-3] + 'jpg'

        image = bob.io.base.load(img_path)
        return image, img_name

    def _get_attribute_baseline_accuracy_val(self, labels, config):
        """
        Get the attribute baseline accuracy (majority guess) on the validation set. The majority guess is based on the
        majority class seen during training
        :param labels: labels
        :param config: evaluation configuration file
        :return: dataframe containing the accuracies of the baseline model (majority guess)
        """

        # Load pickle file containing the majority values from training
        train_attribute_baseline_majority_value = pd.read_pickle(config.dataset.majority_class_file,
                                                                 compression='zip')

        x = labels.apply(pd.Series.value_counts)
        lst = train_attribute_baseline_majority_value.tolist()
        access_tuple_list = [(0 if lst[y] == -1 else 1, y) for y in range(0, len(lst))]
        result_list = []
        for t in access_tuple_list:
            result_list.append((train_attribute_baseline_majority_value.keys()[t[1]], x.iloc[t] / labels.shape[0]))
        return pd.DataFrame(result_list).set_index(0)[1]