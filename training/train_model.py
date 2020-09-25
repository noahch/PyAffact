import copy
import time

import torch
from torch import optim, nn
from torch.optim import lr_scheduler

from network.resnet_51 import resnet50
from preprocessing.dataset_utils import get_train_val_dataset


class TrainModel():
    def __init__(self, config, device):
        self.config = config
        self.device = device
        self.datasets = get_train_val_dataset(config)
        self.model = self._get_model()
        self.model_device = self.model.to(self.device)
        self.optimizer = self._get_optimizer()
        self.criterion = self._get_criterion()
        self.lr_scheduler = self._get_lr_scheduler()

    def train(self):
        if self.config.basic.model == "resnet_51":
            return self._train_resnet_51()
        else:
            raise Exception("Model {} does not have a training routine".format(self.config.basic.model))

    def _get_model(self):
        if self.config.basic.model == "resnet_51":
            return resnet50(pretrained=bool(self.config.basic.pretrained))
        raise Exception("Model {} does not exist".format(self.config.basic.model))

    def _get_optimizer(self):
        if self.config.training.optimizer.type == "SGD":
            return optim.SGD(self.model_device.parameters(),
                                       lr=self.config.training.optimizer.learning_rate,
                                       momentum=self.config.training.optimizer.momentum)
        raise Exception("Optimizer {} does not exist".format(self.config.training.optimizer.type))

    def _get_criterion(self):
        if self.config.training.criterion.type == "BCEWithLogitsLoss":
            return nn.BCEWithLogitsLoss()
        raise Exception("Criterion {} does not exist".format(self.config.training.criterion.type))

    def _get_lr_scheduler(self):
        if self.config.training.lr_scheduler.type == "StepLR":
            return lr_scheduler.StepLR(self.optimizer,
                                       step_size=self.config.training.lr_scheduler.step_size,
                                       gamma=self.config.training.lr_scheduler.gamma)
        raise Exception("Scheduler {} does not exist".format(self.config.training.lr_scheduler.type))

    def _train_resnet_51(self):
        since = time.time()

        best_model_wts = copy.deepcopy(self.model.state_dict())
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

                # Iterate over data.
                for inputs, labels in self.datasets['dataloaders'][phase]:
                    inputs = inputs.to(self.device)
                    labels = labels.to(self.device)

                    # zero the parameter gradients
                    self.optimizer.zero_grad()

                    # forward
                    # track history if only in train
                    with torch.set_grad_enabled(phase == 'train'):
                        outputs = self.model(inputs)
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

        time_elapsed = time.time() - since
        print('Training complete in {:.0f}m {:.0f}s'.format(
            time_elapsed // 60, time_elapsed % 60))
        print('Best val Acc: {:4f}'.format(best_acc))

        # load best model weights
        self.model.load_state_dict(best_model_wts)
        torch.save({
            # 'epoch': EPOCH,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            # 'loss': LOSS,
        }, 'latest_model.pt')
        return self.model

