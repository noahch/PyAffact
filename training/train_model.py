import copy
import os
import time

import torch
from torch.optim import lr_scheduler

from config.config_utils import save_config_to_file
from evaluation.charts import generate_attribute_accuracy_plot
from preprocessing.utils import get_train_val_dataset
from training.model_manager import ModelManager
from utils.utils import get_gpu_memory_map
import wandb
import logging
from tqdm import tqdm


# TODO: Extend from base class
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
        # same training routine for our network
        elif self.config.basic.model == "resnet_noahAndYves":
            return self._train_resnet_51()
        else:
            raise Exception("Model {} does not have a training routine".format(self.config.basic.model))

    def _get_lr_scheduler(self):
        if self.config.training.lr_scheduler.type == "StepLR":
            return lr_scheduler.StepLR(self.optimizer,
                                       step_size=self.config.training.lr_scheduler.step_size,
                                       gamma=self.config.training.lr_scheduler.gamma)
        elif self.config.training.lr_scheduler.type == "ReduceLROnPlateau":
            return lr_scheduler.ReduceLROnPlateau(self.optimizer,
                                       patience=self.config.training.lr_scheduler.patience,
                                       factor=self.config.training.lr_scheduler.gamma, cooldown=1)
        raise Exception("Scheduler {} does not exist".format(self.config.training.lr_scheduler.type))

    def _save_model(self, model_state_dict, optimizer_state_dict, filename):
        torch.save({
            'model_state_dict': model_state_dict,
            'optimizer_state_dict': optimizer_state_dict
        }, os.path.join(self.config.basic.result_directory, filename))

    def _train_resnet_51(self):
        if self.config.basic.enable_wand_reporting:
            wandb.init(project="pyaffact_uzh", entity="uzh", name=self.config.basic.result_directory_name, notes=self.config.basic.experiment_description, config=self.config.toDict())

            wandb.watch(self.model_device)

        since = time.time()

        best_model_wts = copy.deepcopy(self.model.state_dict())
        best_opt_wts = copy.deepcopy(self.optimizer.state_dict())
        best_epoch = ''
        best_acc = 0.0
        # print('Experiment Config')
        # self.config.pprint(pformat='json')
        # print('-' * 50)
        epoch_accuracy_loss_dict = {
            'train': {
                'accuracy': None,
                'loss': None,
            },
            'val': {
                'accuracy': None,
                'loss': None,
            }
        }

        epoch_accuracy_store_dict = {
            'train': None,
            'val': None
        }

        epoch_attributes_correct_count_dict = {
            'train': None,
            'val': None
        }

        # epoch_accuracy_store = None

        # epoch_accuracy_store = torch.zeros(1,40)
        # epoch_accuracy_store = epoch_accuracy_store.to(self.device)
        for epoch in range(self.config.training.epochs):
            # print('Epoch {}/{}'.format(epoch + 1, self.config.training.epochs))
            logging.info('Epoch {}/{}'.format(epoch + 1, self.config.training.epochs))
            # print('-' * 10)


            # Each epoch has a training and validation phase
            for phase in ['train', 'val']:
                if phase == 'train':
                    self.model.train()  # Set model to training mode
                else:
                    self.model.eval()  # Set model to evaluate mode

                attributes_correct_count = torch.zeros(self.datasets['dataset_meta_information']['number_of_labels'])
                attributes_correct_count = attributes_correct_count.to(self.device)
                epoch_attributes_correct_count_dict[phase] = attributes_correct_count
                running_loss = 0.0
                # running_diff = 0.0
                correct_classifications = 0
                get_gpu_memory_map('Before loading input')
                # pbar = tqdm(range(self.datasets['dataset_sizes'][phase]))



                #logging.info(self.datasets['dataloaders']['train'][0][0])

                # assert that inputs shape is of form [batch size, 3, x, y]


                # Iterate over data.
                for inputs, labels, _ in self.datasets['dataloaders'][phase]:

                    # logging.info(inputs.shape)
                    # assert (inputs.shape == (
                    #     self.config.preprocessing.dataloader.batch_size, 3,
                    #     self.config.preprocessing.transformation.crop_size.x,
                    #     self.config.preprocessing.transformation.crop_size.y))
                    # pbar.update(n=inputs.shape[0])
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
                        # prediction_loss = abs(labels.type_as(outputs) - outputs)
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
                    # running_diff += torch.sum(prediction_loss)

                    epoch_attributes_correct_count_dict[phase] += torch.sum(outputs == labels.type_as(outputs), dim=0)
                    correct_classifications += torch.sum(outputs == labels.type_as(outputs))
                    # print(correct_classifications)

                if phase == 'train':
                    if self.config.training.lr_scheduler.type == "ReduceLROnPlateau":
                        self.lr_scheduler.step(loss) # TODO change to val loss
                        if self.lr_scheduler.in_cooldown:
                            logging.info("Changed learning rate from {} to {}".format((1 / self.config.training.lr_scheduler.gamma) * self.optimizer.param_groups[0]["lr"],  self.optimizer.param_groups[0]["lr"]))
                    else:
                        self.lr_scheduler.step()


                epoch_loss = running_loss / self.datasets['dataset_sizes'][phase]
                epoch_accuracy = correct_classifications.double() / (self.datasets['dataset_sizes'][phase] * self.datasets['dataset_meta_information']['number_of_labels'])
                # epoch_acc = running_diff / (self.datasets['dataset_sizes'][phase] * 40)
                epoch_attributes_correct_count_dict[phase] = epoch_attributes_correct_count_dict[phase] / self.datasets['dataset_sizes'][phase]

                epoch_accuracy_loss_dict[phase]['accuracy'] = epoch_accuracy
                epoch_accuracy_loss_dict[phase]['loss'] = epoch_loss

                # if self.config.basic.enable_wand_reporting:
                #     wandb.log({"Accuracy {}".format(phase): epoch_accuracy, "Loss {}".format(phase): epoch_loss, "Baseline": })

                logging.info('{} Loss: {:.4f} \t Acc: {:.4f}'.format(phase, epoch_loss, epoch_accuracy))

                # deep copy the model
                if phase == 'val' and epoch_accuracy > best_acc:
                    best_acc = epoch_accuracy
                    best_model_wts = copy.deepcopy(self.model.state_dict())
                    best_opt_wts = copy.deepcopy(self.optimizer.state_dict())
                    best_epoch = epoch + 1

                if phase == 'val' and (epoch + 1) % self.config.training.save_frequency == 0:
                    self._save_model(copy.deepcopy(self.model.state_dict()), copy.deepcopy(self.optimizer.state_dict()),
                                     '{:03d}.pt'.format(epoch + 1))

            # Generating accuracy for each attribute to generate an epoch accuracy store in the form of [[acc_attr1, 2...], [epoch2_acc_attr1, 2...]]
            # [[acc_attr1, 2...], [epoch2_acc_attr1, 2...]] --> transpose [[epoch1_acc_attr1, epoch2_acc_attr1...], [...]] to generate an per attribute accuracy progress
            if self.config.basic.enable_wand_reporting:
                wandb.log({
                           "Accuracy Train": epoch_accuracy_loss_dict['train']['accuracy'],
                           "Accuracy Val": epoch_accuracy_loss_dict['val']['accuracy'],
                           "Loss Train": epoch_accuracy_loss_dict['train']['loss'],
                           "Loss Val": epoch_accuracy_loss_dict['val']['loss'],
                           "Baseline Train": self.datasets['attribute_baseline_accuracy']['train'].mean(),
                           "Baseline Val": self.datasets['attribute_baseline_accuracy']['val'].mean()
                           }, step=epoch)

                if epoch_accuracy_store_dict['train'] is not None:
                    epoch_accuracy_store_dict['train'] = torch.cat((epoch_accuracy_store_dict['train'], epoch_attributes_correct_count_dict['train'].unsqueeze(0)))
                    epoch_accuracy_store_dict['val'] = torch.cat((epoch_accuracy_store_dict['val'], epoch_attributes_correct_count_dict['val'].unsqueeze(0)))
                    transposed_train = torch.transpose(epoch_accuracy_store_dict['train'], 0, 1).cpu()
                    transposed_val = torch.transpose(epoch_accuracy_store_dict['val'], 0, 1).cpu()
                    for i in range(0, self.datasets['dataset_meta_information']['number_of_labels']):
                        attr_name = self.datasets['dataset_meta_information']['label_names'][i]
                        fig = generate_attribute_accuracy_plot(attr_name,
                                                               transposed_train[i].tolist(),
                                                               self.datasets['attribute_baseline_accuracy']['train'][attr_name],
                                                               transposed_val[i].tolist(),
                                                               self.datasets['attribute_baseline_accuracy']['val'][attr_name],
                                                               )
                        wandb.log({'Accuracy {}'.format(attr_name): fig}, step=epoch)
                else:
                    epoch_accuracy_store_dict['train'] = epoch_attributes_correct_count_dict['train'].unsqueeze(0).clone()
                    epoch_accuracy_store_dict['val'] = epoch_attributes_correct_count_dict['val'].unsqueeze(0).clone()

        # print('Experiment Config')
        # self.config.pprint(pformat='json')
        # print('-' * 50)
        # wandb.log(self.config.pprint(pformat='json'))
        time_elapsed = time.time() - since
        logging.info('Training complete in {:.0f}m {:.0f}s'.format(
            time_elapsed // 60, time_elapsed % 60))
        logging.info('Best val Acc: {:4f}'.format(best_acc))

        self.model.load_state_dict(best_model_wts)
        self._save_model(best_model_wts, best_opt_wts, 'best-{}.pt'.format(best_epoch))
        self.config.evaluation.model_weights_filename = 'best-{}.pt'.format(best_epoch)
        save_config_to_file(self.config)

        self._save_model(copy.deepcopy(self.model.state_dict()), copy.deepcopy(self.optimizer.state_dict()),
                         'latest.pt')
        if self.config.basic.enable_wand_reporting:
            torch.save(self.model_device.state_dict(), os.path.join(wandb.run.dir, 'model_wand.pt'))
        return self.model
