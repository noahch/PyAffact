import copy
import os
import time

import torch
from torch import optim, nn
from torch.optim import lr_scheduler

from network.resnet_51 import resnet50
from preprocessing.utils import get_train_val_dataset
from utils.model_manager import ModelManager
from utils.utils import get_gpu_memory_map


class TrainModel(ModelManager):
    def __init__(self, config, device):
        super().__init__(config, device)
        self.datasets = get_train_val_dataset(config)
        self.optimizer = self._get_optimizer()
        self.criterion = self._get_criterion()
        self.lr_scheduler = self._get_lr_scheduler()
        get_gpu_memory_map('After Model full init')

    def train(self):
        if self.config.basic.model == "resnet_51":
            return self._train_resnet_51()
        else:
            raise Exception("Model {} does not have a training routine".format(self.config.basic.model))


    def _get_lr_scheduler(self):
        if self.config.training.lr_scheduler.type == "StepLR":
            return lr_scheduler.StepLR(self.optimizer,
                                       step_size=self.config.training.lr_scheduler.step_size,
                                       gamma=self.config.training.lr_scheduler.gamma)
        raise Exception("Scheduler {} does not exist".format(self.config.training.lr_scheduler.type))

    def _save_model(self, model_state_dict, optimizer_state_dict, filename):
        torch.save({
            'model_state_dict': model_state_dict,
            'optimizer_state_dict': optimizer_state_dict
        }, os.path.join(self.config.basic.result_directory, filename))

    def _train_resnet_51(self):
        since = time.time()

        best_model_wts = copy.deepcopy(self.model.state_dict())
        best_opt_wts = copy.deepcopy(self.optimizer.state_dict())
        best_epoch = ''
        best_acc = 0.0

        for epoch in range(self.config.training.epochs):
            print('Epoch {}/{}'.format(epoch + 1, self.config.training.epochs))
            print('-' * 10)
            # Each epoch has a training and validation phase
            for phase in ['train', 'val']:
                if phase == 'train':
                    self.model.train()  # Set model to training mode
                else:
                    self.model.eval()  # Set model to evaluate mode

                running_loss = 0.0
                running_diff = 0.0
                correct_classifications = 0
                get_gpu_memory_map('Before loading input')
                # Iterate over data.
                for inputs, labels in self.datasets['dataloaders'][phase]:
                    inputs = inputs.to(self.device)
                    labels = labels.to(self.device)
                    get_gpu_memory_map('After loading input')
                    # zero the parameter gradients
                    self.optimizer.zero_grad()

                    # forward
                    # track history if only in train
                    with torch.set_grad_enabled(phase == 'train'):
                        get_gpu_memory_map('Before loading output')
                        outputs = self.model(inputs)
                        get_gpu_memory_map('After loading output')
                        # print(type(outputs[0][0]))
                        # print(labels.shape)
                        # _, preds = torch.max(outputs, 1)
                        prediction_loss = abs(labels.type_as(outputs) - outputs)
                        loss = self.criterion(outputs, labels.type_as(outputs))
                        # loss = criterion(outputs, torch.max(labels, 1)[1])

                        # backward + optimize only if in training phase
                        if phase == 'train':
                            loss.backward()
                            self.optimizer.step()

                    # statistics
                    running_loss += loss.item() * inputs.size(0)
                    # running_diff += torch.sum(prediction_loss)
                    # outputs_copy = copy.deepcopy(outputs)
                    outputs[outputs < 0.5] = 0
                    outputs[outputs >= 0.5] = 1
                    running_diff += torch.sum(prediction_loss)
                    correct_classifications += torch.sum(outputs == labels.type_as(outputs))
                    # print(running_correct)

                if phase == 'train':
                    self.lr_scheduler.step()

                epoch_loss = running_loss / self.datasets['dataset_sizes'][phase]
                epoch_corr_class = correct_classifications.double() / (self.datasets['dataset_sizes'][phase] * 40)
                epoch_acc = running_diff / (self.datasets['dataset_sizes'][phase] * 40)
                print('{} Loss: {:.4f} \t Correct Classifications: {:.4f} \t AbsLoss: {:.4f}'.format(
                    phase, epoch_loss, epoch_corr_class, epoch_acc))

                # deep copy the model
                if phase == 'val' and epoch_acc > best_acc:
                    best_acc = epoch_acc
                    best_model_wts = copy.deepcopy(self.model.state_dict())
                    best_opt_wts = copy.deepcopy(self.optimizer.state_dict())
                    best_epoch = epoch+1

                if phase == 'val' and (epoch+1) % self.config.training.save_frequency == 0:
                    self._save_model(copy.deepcopy(self.model.state_dict()), copy.deepcopy(self.optimizer.state_dict()), '{:03d}.pt'.format(epoch+1))


        time_elapsed = time.time() - since
        print('Training complete in {:.0f}m {:.0f}s'.format(
            time_elapsed // 60, time_elapsed % 60))
        print('Best val Acc: {:4f}'.format(best_acc))

        self.model.load_state_dict(best_model_wts)
        self._save_model(best_model_wts, best_opt_wts, 'best-{}.pt'.format(best_epoch))
        self._save_model(copy.deepcopy(self.model.state_dict()), copy.deepcopy(self.optimizer.state_dict()),
                         'latest.pt')
        return self.model

