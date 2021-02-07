"""
Model Manager Class which helps setting up the model for training
"""
import torch
from torch import nn
from network.resnet_51 import resnet51
from network.affact_ext import affact_ext
from network.resnet_152 import resnet152


class ModelManager():
    """
    Model Manager Class
    """

    def __init__(self, config, device):
        """
        Init Model Manager
        :param config: DotMap Configuration
        :param device: cuda device
        """
        self.config = config
        self.device = device

        # Get model for training on multiple GPUs
        self.model = nn.DataParallel(self._get_model(), device_ids=[int(
            x[-1]) for x in self.config.basic.cuda_device_name.split(',')])
        if self.config.model.affact_weights and self.config.model.name == 'affact_ext':
            state_dict = torch.load(self.config.model.affact_weights, map_location=self.config.basic.cuda_device_name.split(',')[0])
            self.model.load_state_dict(state_dict['model_state_dict'], strict=False)


        # Transfer model to GPU
        self.model_device = self.model.to(self.device)

    def _get_model(self):
        """
        Get the model based on configuration Value

        :return: A model
        """

        # ResNet-51 used for baseline and AFFACT experiments
        if self.config.model.name == "resnet_51":
            return resnet51(pretrained=bool(self.config.model.pretrained))

        # ResNet-51 used for baseline and AFFACT experiments
        if self.config.model.name == "resnet_152":
            return resnet152(pretrained=bool(self.config.model.pretrained))

        # ResNet-51-ext which extends the Resnet-51 with additional layers
        if self.config.model.name == "affact_ext":
            return affact_ext(pretrained=bool(self.config.model.pretrained), dropout=self.config.model.dropout)

        raise Exception("Model {} does not exist".format(self.config.model.name))


