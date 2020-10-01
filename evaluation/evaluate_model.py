from preprocessing.utils import get_train_val_dataset
import os
import matplotlib.pyplot as plt
import numpy as np
import torch

from preprocessing.utils import get_train_val_dataset
from utils.model_manager import ModelManager


class EvalModel(ModelManager):
    def __init__(self, config, device):
        super().__init__(config, device)
        self.datasets = get_train_val_dataset(config)
        self.optimizer = self._get_optimizer()


    def eval(self):
        checkpoint = torch.load(os.path.join(self.config.basic.result_directory, 'latest.pt'))
        self.model_device.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.visualize_model(self.model_device, 6)


    def imshow(self, inp, title=None):
        """Imshow for Tensor."""
        inp = inp.numpy().transpose((1, 2, 0))
        # mean = np.array([0.485, 0.456, 0.406])
        # std = np.array([0.229, 0.224, 0.225])
        # inp = std * inp + mean
        inp = np.clip(inp, 0, 1)
        plt.imshow(inp)
        if title is not None:
            plt.title(title)
        plt.pause(0.001)  # pause a bit so that plots are updated

    def visualize_model(self, model, num_images=6):
        was_training = model.training
        model.eval()
        images_so_far = 0
        fig = plt.figure()

        with torch.no_grad():
            for i, (inputs, labels) in enumerate(self.datasets['dataloaders']['val']):
                inputs = inputs.to(self.device)
                labels = labels.to(self.device)

                outputs = model(inputs)

                # _, preds = torch.max(outputs, 1)

                for j in range(inputs.size()[0]):
                    black = outputs[j][8]
                    blonde = outputs[j][9]
                    brown = outputs[j][10]
                    images_so_far += 1
                    ax = plt.subplot(num_images//2, 2, images_so_far)
                    ax.axis('off')
                    ax.set_title('Black: {:.2f}  blonde: {:.2f} brown: {:.2f}'.format(black, blonde, brown))
                    self.imshow(inputs.cpu().data[j])

                    if images_so_far == num_images:
                        model.train(mode=was_training)
                        return
            model.train(mode=was_training)
