"""
Class that handles training of the model
"""
import copy
import os
import time

import torch
from torch import optim, nn
from torch.optim import lr_scheduler

from evaluation.charts import generate_attribute_accuracy_plot
from preprocessing.dataset_generator import get_train_val_dataset
from training.model_manager import ModelManager
import wandb
import logging
from tqdm import tqdm


class TrainModel(ModelManager):
    """
    Class that manages the training of models
    """

    def __init__(self, config, device):
        """
        init
        :param config: the training configuration file
        :param device: the device
        """
        super().__init__(config, device)

        # get the training an validation datasets
        self.datasets = get_train_val_dataset(config)
        # get the optimizer
        self.optimizer = self._get_optimizer()
        # get the loss criterion
        self.criterion = self._get_criterion()
        # get the learning rate scheduler
        self.lr_scheduler = self._get_lr_scheduler()

    def train(self):
        """
        train the model
        :return: result of a specific training process (trained model)
        """

        # train the resnet_51 (ResNet-51-S, AFFACT)
        if self.config.model.name == "resnet_51":
            return self._train_resnet_51()

        # train resnet_152 (experiment)
        elif self.config.model.name == "resnet_152":
            return self._train_resnet_51()

        # train extended resnet_51 (experiment)
        elif self.config.model.name == "resnet_51-ext":
            return self._train_resnet_51()

        else:
            raise Exception("Model {} does not have a training routine".format(self.config.model.name))

    def _get_lr_scheduler(self):
        """
        get the learning rate scheduler
        :return: LR Scheduler
        """

        # Step learning rate
        if self.config.training.lr_scheduler.type == "StepLR":
            return lr_scheduler.StepLR(self.optimizer,
                                       step_size=self.config.training.lr_scheduler.step_size,
                                       gamma=self.config.training.lr_scheduler.gamma)

        # Reduce on plateau learning rate
        elif self.config.training.lr_scheduler.type == "ReduceLROnPlateau":
            return lr_scheduler.ReduceLROnPlateau(self.optimizer,
                                                  patience=self.config.training.lr_scheduler.patience,
                                                  factor=self.config.training.lr_scheduler.gamma, cooldown=1)
        raise Exception("Scheduler {} does not exist".format(self.config.training.lr_scheduler.type))

    def _save_model(self, model_state_dict, optimizer_state_dict, filename):
        """
        Saves model and optimizer
        :param model_state_dict: weights and bias of model
        :param optimizer_state_dict: weight of optimizer
        :param filename: name of the model
        """
        torch.save({
            'model_state_dict': model_state_dict,
            'optimizer_state_dict': optimizer_state_dict
        }, os.path.join(self.config.basic.result_directory, filename))

    def _get_optimizer(self):
        """
        Get different optimizers for different experiments
        :return: Optimizer
        """

        # SGD Optimizer
        if self.config.training.optimizer.type == "SGD":
            return optim.SGD(self.model_device.parameters(),
                             lr=self.config.training.optimizer.learning_rate,
                             momentum=self.config.training.optimizer.momentum)

        # RMSprop Optimizer
        if self.config.training.optimizer.type == "RMSprop":
            return optim.RMSprop(self.model_device.parameters(),
                                 lr=self.config.training.optimizer.learning_rate,
                                 momentum=self.config.training.optimizer.momentum)

        # Adam Optimizer
        if self.config.training.optimizer.type == "Adam":
            return optim.Adam(self.model_device.parameters(),
                              lr=self.config.training.optimizer.learning_rate)

        raise Exception("Optimizer {} does not exist".format(self.config.training.optimizer.type))

    def _get_criterion(self):
        """
        Get different criterions for different experiments
        :return: Loss Function
        """

        # Binary Cross Entropy with Logits Loss
        if self.config.training.criterion.type == "BCEWithLogitsLoss":
            return nn.BCEWithLogitsLoss()

        # Binary Cross Entropy Loss
        if self.config.training.criterion.type == "BCELoss":
            return nn.BCELoss()

        raise Exception("Criterion {} does not exist".format(self.config.training.criterion.type))

    def _train_resnet_51(self):
        """
        Training Loop for the resnet_51 architecture. Can also be used for the resnet_152 or the extended resnet_51
        :return: the trained model
        """

        # Initialize WandB Reporting if enabled
        if self.config.basic.enable_wand_reporting:
            wandb.init(project="pyaffact_uzh", entity="uzh", name=self.config.basic.result_directory_name,
                       notes=self.config.basic.experiment_description, config=self.config.toDict())

            wandb.watch(self.model_device)

        since = time.time()

        # Structures to save best performing model and optimizer weights
        best_model_wts = copy.deepcopy(self.model.state_dict())
        best_opt_wts = copy.deepcopy(self.optimizer.state_dict())
        best_epoch = ''
        best_acc = 0.0

        # Structure to keep track of epoch accuracies and losses of each phase
        epoch_metrics_dict = {
            'train': {
                'accuracy': None,
                'loss': None,
            },
            'val': {
                'accuracy': None,
                'loss': None,
            }
        }

        # Structure to store attribute accuracy per epoch for each phase (used by wandb)
        epoch_per_attribute_accuracy_wandb_dict = {
            'train': None,
            'val': None
        }

        # Structure to count number of correct classifications per epoch for each phase
        epoch_attributes_correct_count_dict = {
            'train': None,
            'val': None
        }

        # Structure to save attribute accuracy per epoch for each phase
        epoch_per_attribute_accuracy_dict = {
            'train': None,
            'val': None
        }
        # Training Loop
        for epoch in range(self.config.training.epochs):

            logging.info('Epoch {}/{}'.format(epoch + 1, self.config.training.epochs))

            # Each epoch has a training and validation phase
            for phase in ['train', 'val']:
                if phase == 'train':
                    # Set model to training mode
                    self.model.train()
                else:
                    # Set model to evaluate mode
                    self.model.eval()

                # Init structures for metrics
                attributes_correct_count = torch.zeros(self.datasets['dataset_meta_information']['number_of_labels'])
                attributes_correct_count = attributes_correct_count.to(self.device)
                epoch_attributes_correct_count_dict[phase] = attributes_correct_count
                running_loss = 0.0
                correct_classifications = 0

                # Initialize progress bar
                progress_bar = tqdm(range(self.datasets['dataset_sizes'][phase]))

                # Iterate over data
                for inputs, labels, _ in self.datasets['dataloaders'][phase]:

                    # Update progress bar
                    progress_bar.update(n=inputs.shape[0])

                    # Transfer input and labels to GPU/Device
                    inputs = inputs.to(self.device)
                    labels = labels.to(self.device)

                    # Zero the parameter gradients
                    self.optimizer.zero_grad()

                    # Forward pass
                    # Only track history if in training phase
                    with torch.set_grad_enabled(phase == 'train'):
                        # Get predictions from model
                        predictions = self.model(inputs)

                        # Calculate the loss
                        loss = self.criterion(predictions, labels.type_as(predictions))

                        # Backward pass and optimizer step, only in training phase
                        if phase == 'train':
                            loss.backward()
                            self.optimizer.step()

                    # Calculate running loss
                    running_loss += loss.item() * inputs.size(0)

                    # Map probability prediction to yes or no
                    predictions[predictions < 0.5] = 0
                    predictions[predictions >= 0.5] = 1

                    # Compare label with prediction, sum up correct classifications per attribute
                    epoch_attributes_correct_count_dict[phase] += torch.sum(predictions == labels.type_as(predictions), dim=0)

                    # Compare label with prediction, sum up correct classifications overall
                    correct_classifications += torch.sum(predictions == labels.type_as(predictions))

                # Learning rate scheduler step if in validation phase
                if phase == 'val':

                    # learning rate scheduler type is reduce learning rate on plateau
                    if self.config.training.lr_scheduler.type == "ReduceLROnPlateau":
                        # LRScheduler step
                        self.lr_scheduler.step(loss)

                        # If the learning rate is reduced
                        if self.lr_scheduler.in_cooldown:
                            logging.info(
                                "Changed learning rate from {} to {}. Reinitializing model weights with best model from epoch {}".format(
                                    (1 / self.config.training.lr_scheduler.gamma) *
                                    self.optimizer.param_groups[0]["lr"],
                                    self.optimizer.param_groups[0]["lr"],
                                    best_epoch))
                            # Reinitialize the model with the previous best weights
                            self.model_device.load_state_dict(best_model_wts)

                    # learning rate scheduler type is step learning rate
                    else:
                        # LRScheduler step
                        self.lr_scheduler.step()

                # Calculate Epoch Loss
                epoch_loss = running_loss / self.datasets['dataset_sizes'][phase]

                # Calculate Epoch Accuracy
                epoch_accuracy = correct_classifications.double(
                ) / (self.datasets['dataset_sizes'][phase] * self.datasets['dataset_meta_information'][
                    'number_of_labels'])

                # Calculate per attribute accuracy
                epoch_per_attribute_accuracy_dict[phase] = epoch_attributes_correct_count_dict[phase] / \
                                                             self.datasets['dataset_sizes'][phase]

                # Save metrics in structure
                epoch_metrics_dict[phase]['accuracy'] = epoch_accuracy
                epoch_metrics_dict[phase]['loss'] = epoch_loss

                logging.info('{} Loss: {:.4f} \t Acc: {:.4f}'.format(phase, epoch_loss, epoch_accuracy))

                # Deep copy the model if new accuracy on validation is better than previous best one
                if phase == 'val' and epoch_accuracy > best_acc:
                    best_acc = epoch_accuracy
                    best_model_wts = copy.deepcopy(self.model.state_dict())
                    best_opt_wts = copy.deepcopy(self.optimizer.state_dict())
                    best_epoch = epoch + 1

                # Save Checkpoint during training in predetermined frequency
                if phase == 'val' and (epoch + 1) % self.config.training.save_frequency == 0:
                    self._save_model(copy.deepcopy(self.model.state_dict()), copy.deepcopy(self.optimizer.state_dict()),
                                     '{:03d}.pt'.format(epoch + 1))

            # Generating accuracy chart for each attribute andto generate an epoch accuracy store in the form of [[acc_attr1, 2...], [epoch2_acc_attr1, 2...]]
            # [[acc_attr1, 2...], [epoch2_acc_attr1, 2...]] --> transpose [[epoch1_acc_attr1, epoch2_acc_attr1...], [...]] to generate an per attribute accuracy progress

            # Report training metrics to wand
            if self.config.basic.enable_wand_reporting:
                # Log current metrics
                wandb.log({
                    "Accuracy Train": epoch_metrics_dict['train']['accuracy'],
                    "Accuracy Val": epoch_metrics_dict['val']['accuracy'],
                    "Loss Train": epoch_metrics_dict['train']['loss'],
                    "Loss Val": epoch_metrics_dict['val']['loss'],
                    "Baseline Train": self.datasets['attribute_baseline_accuracy']['train'].mean(),
                    "Baseline Val": self.datasets['attribute_baseline_accuracy']['val'].mean()
                }, step=epoch)

                if epoch_per_attribute_accuracy_wandb_dict['train'] is not None:
                    # Prepare data structures for logging to wandb
                    epoch_per_attribute_accuracy_wandb_dict['train'] = torch.cat(
                        (epoch_per_attribute_accuracy_wandb_dict['train'], epoch_per_attribute_accuracy_dict['train'].unsqueeze(0)))
                    epoch_per_attribute_accuracy_wandb_dict['val'] = torch.cat(
                        (epoch_per_attribute_accuracy_wandb_dict['val'], epoch_per_attribute_accuracy_dict['val'].unsqueeze(0)))
                    transposed_train = torch.transpose(epoch_per_attribute_accuracy_wandb_dict['train'], 0, 1).cpu()
                    transposed_val = torch.transpose(epoch_per_attribute_accuracy_wandb_dict['val'], 0, 1).cpu()

                    # For each attribute, generate accuracy chart over epochs and log it to wandb
                    for i in range(0, self.datasets['dataset_meta_information']['number_of_labels']):
                        attr_name = self.datasets['dataset_meta_information']['label_names'][i]
                        fig = generate_attribute_accuracy_plot(attr_name,
                                                               transposed_train[i].tolist(),
                                                               self.datasets['attribute_baseline_accuracy']['train'][
                                                                   attr_name],
                                                               transposed_val[i].tolist(),
                                                               self.datasets['attribute_baseline_accuracy']['val'][
                                                                   attr_name],
                                                               )
                        wandb.log({'Accuracy {}'.format(attr_name): fig}, step=epoch)
                else:
                    # Prepare data structures for logging to wandb
                    epoch_per_attribute_accuracy_wandb_dict['train'] = epoch_per_attribute_accuracy_dict['train'].unsqueeze(
                        0).clone()
                    epoch_per_attribute_accuracy_wandb_dict['val'] = epoch_per_attribute_accuracy_dict['val'].unsqueeze(0).clone()

            # Close progress bar
            progress_bar.close()

        # Log training metrics
        time_elapsed = time.time() - since
        logging.info('Training complete in {:.0f}m {:.0f}s'.format(
            time_elapsed // 60, time_elapsed % 60))
        logging.info('Best val Acc: {:4f}'.format(best_acc))

        # Save the best model weights
        self._save_model(best_model_wts, best_opt_wts, 'best-{}.pt'.format(best_epoch))

        # Save latest model
        self._save_model(copy.deepcopy(self.model.state_dict()), copy.deepcopy(self.optimizer.state_dict()),
                         'latest.pt')

        # Reinitialize model with best weights during training and prepare it for returning
        self.model.load_state_dict(best_model_wts)

        # TODO: Check if needed
        if self.config.basic.enable_wand_reporting:
            torch.save(self.model_device.state_dict(), os.path.join(wandb.run.dir, 'model_wand.pt'))

        # Return model
        return self.model
