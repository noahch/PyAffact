import copy
import os
import time

import matplotlib
import matplotlib.pyplot as plt
import torch
import numpy as np
from PIL import Image
from torch import optim, nn
from torch.optim import lr_scheduler

from preprocessing.utils import get_train_val_dataset
from utils.model_manager import ModelManager
from utils.utils import get_gpu_memory_map
from torchvision.transforms import transforms

class TrainModel(ModelManager):
    def __init__(self, config, device):
        super().__init__(config, device)
        self.datasets = get_train_val_dataset(config)
        print(self.model)
        self.optimizer = self._get_optimizer()
        self.criterion = self._get_criterion()
        self.lr_scheduler = self._get_lr_scheduler()
        get_gpu_memory_map('After Model full init')

    def train(self):
        if self.config.basic.model == "auto_encoder":
            return self._train_auto_encoder()
        elif self.config.basic.model == "auto_encoder_flat":
            return self._train_auto_encoder()
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

    def _train_auto_encoder(self):
        matplotlib.use('Agg')
        best_model_wts = copy.deepcopy(self.model.state_dict())
        best_opt_wts = copy.deepcopy(self.optimizer.state_dict())
        best_epoch = ''
        best_loss = 0.0
        print('Experiment Config')
        self.config.pprint(pformat='json')
        print('-' * 50)

        for epoch in range(self.config.training.epochs):
            # monitor training loss
            self.lr_scheduler.step(epoch)
            print(self.optimizer.param_groups[0]['lr'])

            for phase in ['train', 'val']:
                train_loss = 0.0
                if phase == 'train':
                    self.model.train()  # Set model to training mode
                else:
                    self.model.eval()  # Set model to evaluate mode
                ###################
                # train the model #
                ###################
                for data in self.datasets['dataloaders'][phase]:
                    # _ stands in for labels, here
                    # no need to flatten images
                    images = data
                    images = images.to(self.device)
                    # clear the gradients of all optimized variables
                    self.optimizer.zero_grad()
                    # forward pass: compute predicted outputs by passing inputs to the model
                    outputs = self.model_device(images)
                    # calculate the loss
                    loss = self.criterion(outputs, images)
                    # backward pass: compute gradient of the loss with respect to model parameters
                    loss.backward()
                    # print(self.model_device.conv5.weight.grad)
                    # perform a single optimization step (parameter update)
                    self.optimizer.step()
                    # update running training loss
                    train_loss += loss.item() * images.size(0)

                    if phase == 'val' and epoch % self.config.training.save_frequency == 0:
                        in_image = images[0]
                        in_image = in_image.detach().cpu().numpy()
                        in_image = np.transpose(in_image, (1, 2, 0))
                        in_image = (in_image * 1 + 0) * 255
                        in_image = in_image.astype(np.uint8)
                        plt.clf()
                        plt.figure(1)
                        plt.subplot(121)
                        plt.imshow(Image.fromarray(in_image, 'RGB'))

                        out_image = self.model_device(images[0].unsqueeze(0))[0]
                        out_image = out_image.detach().cpu().numpy()
                        out_image = np.transpose(out_image, (1, 2, 0))
                        out_image = (out_image * 1 + 0) * 255
                        out_image = out_image.astype(np.uint8)
                        plt.subplot(122)
                        plt.imshow(Image.fromarray(out_image, 'RGB'))
                        plt.savefig(os.path.join(self.config.basic.result_directory, '{}.jpg'.format(epoch)))


                # print avg training statistics
                train_loss = train_loss / len(self.datasets['dataloaders'][phase])
                print('Epoch: {} \t{} Loss: {:.6f}'.format(
                    epoch,
                    phase,
                    train_loss
                ))

                # deep copy the model
                if phase == 'val' and train_loss < best_loss:
                    best_loss = train_loss
                    best_model_wts = copy.deepcopy(self.model.state_dict())
                    best_opt_wts = copy.deepcopy(self.optimizer.state_dict())
                    best_epoch = epoch + 1

                if phase == 'val' and (epoch + 1) % self.config.training.save_frequency == 0:
                    self._save_model(copy.deepcopy(self.model.state_dict()), copy.deepcopy(self.optimizer.state_dict()),
                                     '{:03d}.pt'.format(epoch + 1))


        self.model.load_state_dict(best_model_wts)
        self._save_model(best_model_wts, best_opt_wts, 'best-{}.pt'.format(best_epoch))
        self._save_model(copy.deepcopy(self.model.state_dict()), copy.deepcopy(self.optimizer.state_dict()),
                         'latest.pt')
        return self.model


