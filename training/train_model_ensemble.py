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

from network.auto_encoder import ConvAutoencoder
from network.auto_encoder_flat import FlatAutoencoder
from preprocessing.utils import get_train_val_dataset, get_train_val_dataset_AB
from utils.model_manager import ModelManager
from utils.utils import get_gpu_memory_map
from torchvision.transforms import transforms
import torch.nn.functional as F
class TrainModelEnsemble():
    def __init__(self, config, device):
        self.config = config
        self.device = device
        self.datasets = get_train_val_dataset_AB(config)
        #
        self.central = nn.Sequential(
            nn.Conv2d(8, 8, kernel_size=3, padding=1, stride=1),
            nn.ReLU(),
            nn.ConvTranspose2d(8, 8, kernel_size=3, stride=2, output_padding=1, padding=1)
        )
        self.model_a = FlatAutoencoder(self.central)
        self.model_device_a = self.model_a.to(self.device)
        self.optimizer_a = optim.Adam(self.model_device_a.parameters(), lr=self.config.training.optimizer.learning_rate)
        self.criterion_a = nn.BCELoss()
        self.lr_scheduler_a = lr_scheduler.StepLR(self.optimizer_a,
                                       step_size=self.config.training.lr_scheduler.step_size,
                                       gamma=self.config.training.lr_scheduler.gamma)
        self.model_b = FlatAutoencoder(self.central)
        self.model_device_b = self.model_b.to(self.device)
        self.optimizer_b = optim.Adam(self.model_device_b.parameters(), lr=self.config.training.optimizer.learning_rate)
        self.criterion_b = nn.BCELoss()
        self.lr_scheduler_b = lr_scheduler.StepLR(self.optimizer_b,
                                       step_size=self.config.training.lr_scheduler.step_size,
                                       gamma=self.config.training.lr_scheduler.gamma)
        get_gpu_memory_map('After Model full init')

    def train(self):
        if self.config.basic.model == "auto_encoder":
            return self._train_auto_encoder()
        elif self.config.basic.model == "auto_encoder_flat":
            return self._train_auto_encoder()
        else:
            raise Exception("Model {} does not have a training routine".format(self.config.basic.model))




    def _save_model(self, model_state_dict, optimizer_state_dict, filename):
        torch.save({
            'model_state_dict': model_state_dict,
            'optimizer_state_dict': optimizer_state_dict
        }, os.path.join(self.config.basic.result_directory, filename))

    def _train_auto_encoder(self):
        matplotlib.use('Agg')
        best_model_wts_a = copy.deepcopy(self.model_a.state_dict())
        best_model_wts_b = copy.deepcopy(self.model_b.state_dict())
        best_opt_wts_a = copy.deepcopy(self.optimizer_a.state_dict())
        best_opt_wts_b = copy.deepcopy(self.optimizer_b.state_dict())
        best_epoch_a = ''
        best_epoch_b = ''
        best_loss_a = 0.0
        best_loss_b = 0.0
        print('Experiment Config')
        self.config.pprint(pformat='json')
        print('-' * 50)

        last_weight_central = torch.Tensor()
        last_init = None

        for epoch in range(self.config.training.epochs):
            # monitor training loss
            self.lr_scheduler_a.step(epoch)
            self.lr_scheduler_b.step(epoch)
            for phase in ['train_a', 'train_b', 'val_a', 'val_b']:
                train_loss = 0.0
                if phase.startswith('train'):
                    self.model_a.train()  # Set model to training mode
                    self.model_b.train()  # Set model to training mode
                else:
                    self.model_a.eval()  # Set model to evaluate mode
                    self.model_b.eval()  # Set model to evaluate mode

                ###################
                # train the model #
                ###################
                for data in self.datasets['dataloaders'][phase]:
                    # _ stands in for labels, here
                    # no need to flatten images
                    images = data
                    images = images.to(self.device)
                    # clear the gradients of all optimized variables
                    if phase == 'train_a' or phase == 'val_a':
                        self.optimizer_a.zero_grad()
                        # forward pass: compute predicted outputs by passing inputs to the model
                        outputs = self.model_device_a(images)
                        # calculate the loss
                        loss = self.criterion_a(outputs, images)
                        # backward pass: compute gradient of the loss with respect to model parameters
                        loss.backward()
                        # perform a single optimization step (parameter update)
                        self.optimizer_a.step()
                        # update running training loss
                        train_loss += loss.item() * images.size(0)

                        if phase == 'val_a' and epoch % self.config.training.save_frequency == 0:
                            in_image = images[0]
                            in_image = in_image.detach().cpu().numpy()
                            in_image = np.transpose(in_image, (1, 2, 0))
                            in_image = (in_image * 1 + 0) * 255
                            in_image = in_image.astype(np.uint8)
                            plt.clf()
                            plt.figure(1)
                            plt.subplot(131)
                            plt.imshow(Image.fromarray(in_image, 'RGB'))

                            out_image = self.model_device_a(images[0].unsqueeze(0))[0]
                            out_image = out_image.detach().cpu().numpy()
                            out_image = np.transpose(out_image, (1, 2, 0))
                            out_image = (out_image * 1 + 0) * 255
                            out_image = out_image.astype(np.uint8)
                            plt.subplot(132)
                            plt.imshow(Image.fromarray(out_image, 'RGB'))

                            pred_model = FlatAutoencoder(self.central)
                            new_state_dict = dict()
                            new_state_dict['conv1.weight'] = self.model_a.state_dict()['conv1.weight']
                            new_state_dict['conv1.bias'] = self.model_a.state_dict()['conv1.bias']
                            new_state_dict['conv2.weight'] = self.model_a.state_dict()['conv2.weight']
                            new_state_dict['conv2.bias'] = self.model_a.state_dict()['conv2.bias']
                            new_state_dict['conv3.weight'] = self.model_a.state_dict()['conv3.weight']
                            new_state_dict['conv3.bias'] = self.model_a.state_dict()['conv3.bias']
                            new_state_dict['conv4.weight'] = self.model_a.state_dict()['conv4.weight']
                            new_state_dict['conv4.bias'] = self.model_a.state_dict()['conv4.bias']
                            new_state_dict['conv5.weight'] = self.model_a.state_dict()['conv5.weight']
                            new_state_dict['conv5.bias'] = self.model_a.state_dict()['conv5.bias']
                            new_state_dict['central.0.weight'] = self.model_a.state_dict()['central.0.weight']
                            new_state_dict['central.0.bias'] = self.model_a.state_dict()['central.0.bias']
                            new_state_dict['central.2.weight'] = self.model_a.state_dict()['central.2.weight']
                            new_state_dict['central.2.bias'] = self.model_a.state_dict()['central.2.bias']
                            new_state_dict['t_conv2.weight'] = self.model_b.state_dict()['t_conv2.weight']
                            new_state_dict['t_conv2.bias'] = self.model_b.state_dict()['t_conv2.bias']
                            new_state_dict['t_conv3.weight'] = self.model_b.state_dict()['t_conv3.weight']
                            new_state_dict['t_conv3.bias'] = self.model_b.state_dict()['t_conv3.bias']
                            new_state_dict['t_conv4.weight'] = self.model_b.state_dict()['t_conv4.weight']
                            new_state_dict['t_conv4.bias'] = self.model_b.state_dict()['t_conv4.bias']
                            pred_model.load_state_dict(new_state_dict)
                            pred_model_device = pred_model.to(self.device)
                            out_image = pred_model_device(images[0].unsqueeze(0))[0]
                            out_image = out_image.detach().cpu().numpy()
                            out_image = np.transpose(out_image, (1, 2, 0))
                            out_image = (out_image * 1 + 0) * 255
                            out_image = out_image.astype(np.uint8)
                            plt.subplot(133)
                            plt.imshow(Image.fromarray(out_image, 'RGB'))

                            plt.savefig(os.path.join(self.config.basic.result_directory, 'a{}.jpg'.format(epoch)))



                    if phase == 'train_b' or phase == 'val_b':
                        self.optimizer_b.zero_grad()
                        # forward pass: compute predicted outputs by passing inputs to the model
                        outputs = self.model_device_b(images)
                        # calculate the loss
                        loss = self.criterion_b(outputs, images)
                        # backward pass: compute gradient of the loss with respect to model parameters
                        loss.backward()
                        # print(self.model_device.conv5.weight.grad)
                        # perform a single optimization step (parameter update)
                        self.optimizer_b.step()
                        # update running training loss
                        train_loss += loss.item() * images.size(0)

                        if phase == 'val_b' and epoch % self.config.training.save_frequency == 0:
                            in_image = images[0]
                            in_image = in_image.detach().cpu().numpy()
                            in_image = np.transpose(in_image, (1, 2, 0))
                            in_image = (in_image * 1 + 0) * 255
                            in_image = in_image.astype(np.uint8)
                            plt.clf()
                            plt.figure(1)
                            plt.subplot(131)
                            plt.imshow(Image.fromarray(in_image, 'RGB'))

                            out_image = self.model_device_b(images[0].unsqueeze(0))[0]
                            out_image = out_image.detach().cpu().numpy()
                            out_image = np.transpose(out_image, (1, 2, 0))
                            out_image = (out_image * 1 + 0) * 255
                            out_image = out_image.astype(np.uint8)
                            plt.subplot(132)
                            plt.imshow(Image.fromarray(out_image, 'RGB'))

                            pred_model = FlatAutoencoder(self.central)
                            new_state_dict = dict()
                            new_state_dict['conv1.weight'] = self.model_b.state_dict()['conv1.weight']
                            new_state_dict['conv1.bias'] = self.model_b.state_dict()['conv1.bias']
                            new_state_dict['conv2.weight'] = self.model_b.state_dict()['conv2.weight']
                            new_state_dict['conv2.bias'] = self.model_b.state_dict()['conv2.bias']
                            new_state_dict['conv3.weight'] = self.model_b.state_dict()['conv3.weight']
                            new_state_dict['conv3.bias'] = self.model_b.state_dict()['conv3.bias']
                            new_state_dict['conv4.weight'] = self.model_b.state_dict()['conv4.weight']
                            new_state_dict['conv4.bias'] = self.model_b.state_dict()['conv4.bias']
                            new_state_dict['conv5.weight'] = self.model_b.state_dict()['conv5.weight']
                            new_state_dict['conv5.bias'] = self.model_b.state_dict()['conv5.bias']
                            new_state_dict['central.0.weight'] = self.model_b.state_dict()['central.0.weight']
                            new_state_dict['central.0.bias'] = self.model_b.state_dict()['central.0.bias']
                            new_state_dict['central.2.weight'] = self.model_b.state_dict()['central.2.weight']
                            new_state_dict['central.2.bias'] = self.model_b.state_dict()['central.2.bias']
                            new_state_dict['t_conv2.weight'] = self.model_a.state_dict()['t_conv2.weight']
                            new_state_dict['t_conv2.bias'] = self.model_a.state_dict()['t_conv2.bias']
                            new_state_dict['t_conv3.weight'] = self.model_a.state_dict()['t_conv3.weight']
                            new_state_dict['t_conv3.bias'] = self.model_a.state_dict()['t_conv3.bias']
                            new_state_dict['t_conv4.weight'] = self.model_a.state_dict()['t_conv4.weight']
                            new_state_dict['t_conv4.bias'] = self.model_a.state_dict()['t_conv4.bias']
                            pred_model.load_state_dict(new_state_dict)
                            pred_model_device = pred_model.to(self.device)
                            out_image = pred_model_device(images[0].unsqueeze(0))[0]
                            out_image = out_image.detach().cpu().numpy()
                            out_image = np.transpose(out_image, (1, 2, 0))
                            out_image = (out_image * 1 + 0) * 255
                            out_image = out_image.astype(np.uint8)
                            plt.subplot(133)
                            plt.imshow(Image.fromarray(out_image, 'RGB'))
                            plt.savefig(os.path.join(self.config.basic.result_directory, 'b{}.jpg'.format(epoch)))
                # print('MODEL A')
                # print(self.model_device_a.central[0].weight[0][0])
                # print('MODEL B')
                # print(self.model_device_b.central[0].weight[0][0])
                #
                # print('MODEL A conv1')
                # print(self.model_device_a.conv1.weight[0][0])
                # print('MODEL B conv1')
                # print(self.model_device_b.conv1.weight[0][0])
                # print(last_weight_central)
                # if last_init and torch.all(torch.eq(last_weight_central, self.model_device_a.central[0].weight)):
                #     print("IS SAME AS LAST")
                # else:
                #     print("IS NOT SAME AS LAST")
                # last_init = True
                # last_weight_central = copy.deepcopy(self.model_device_a.central[0].weight)
                #
                #
                # if torch.all(torch.eq(self.model_device_a.central[0].weight, self.model_device_b.central[0].weight)):
                #     print("IS SAME")
                # else:
                #     print("IS NOT SAME")


                # print avg training statistics
                train_loss = train_loss / len(self.datasets['dataloaders'][phase])
                print('Epoch: {} \t{} Loss: {:.6f}'.format(
                    epoch,
                    phase,
                    train_loss
                ))

                # deep copy the model
                if phase == 'val_a' and train_loss < best_loss_a:
                    best_loss_a = train_loss
                    best_model_wts_a = copy.deepcopy(self.model_a.state_dict())
                    best_opt_wts_a = copy.deepcopy(self.optimizer_a.state_dict())
                    best_epoch_a = epoch + 1
                # deep copy the model
                if phase == 'val_b' and train_loss < best_loss_b:
                    best_loss_b = train_loss
                    best_model_wts_b = copy.deepcopy(self.model_b.state_dict())
                    best_opt_wts_b = copy.deepcopy(self.optimizer_b.state_dict())
                    best_epoch_b = epoch + 1

                if phase == 'val_a' and (epoch + 1) % self.config.training.save_frequency == 0:
                    self._save_model(copy.deepcopy(self.model_a.state_dict()), copy.deepcopy(self.optimizer_a.state_dict()),
                                     'a{:03d}.pt'.format(epoch + 1))
                if phase == 'val_b' and (epoch + 1) % self.config.training.save_frequency == 0:
                    self._save_model(copy.deepcopy(self.model_b.state_dict()), copy.deepcopy(self.optimizer_b.state_dict()),
                                     'b{:03d}.pt'.format(epoch + 1))
            # self.lr_scheduler_a.step(epoch)
            # self.lr_scheduler_b.step(epoch)
            print('LR Both: {}'.format(self.optimizer_a.param_groups[0]['lr']))
            # print('LR B: {}'.format(self.optimizer_b.param_groups[0]['lr']))

        self.model_a.load_state_dict(best_model_wts_a)
        self.model_b.load_state_dict(best_model_wts_b)
        self._save_model(best_model_wts_a, best_opt_wts_a, 'best-a-{}.pt'.format(best_epoch_a))
        self._save_model(best_model_wts_b, best_opt_wts_b, 'best-b-{}.pt'.format(best_epoch_b))
        self._save_model(copy.deepcopy(self.model_a.state_dict()), copy.deepcopy(self.optimizer_a.state_dict()),
                         'latest-a.pt')
        self._save_model(copy.deepcopy(self.model_b.state_dict()), copy.deepcopy(self.optimizer_b.state_dict()),
                         'latest-b.pt')
        return self.model_a, self.model_b
        # return self.model_a


