"""
Model Manager Class which helps setting up the model for training
"""
from torch import optim, nn
from network.resnet_51 import resnet51
from network.resnet_51_ext import resnet51_ext


class ModelManager():
    """
    Model Manager Class
    """

    def __init__(self, config, device):
        """
        Init Model Manager

        Parameters
        ----------
        config DotMap Configuration
        device cuda device
        """
        self.config = config
        self.device = device

        # Get model for training on multiple GPUs
        if len(self.config.basic.cuda_device_name.split(',')) > 1:
            self.model = nn.DataParallel(self._get_model(), device_ids=[int(
                x[-1]) for x in self.config.basic.cuda_device_name.split(',')])
        # Get model for training on one GPU
        else:
            self.model = self._get_model()

        # Transfer model to GPU
        self.model_device = self.model.to(self.device)

    def _get_model(self):
        """
        Get the model based on configuration Value

        Returns
        -------
        A model
        """

        # ResNet-51 used for baseline and AFFACT experiments
        if self.config.basic.model == "resnet_51":
            return resnet51(pretrained=bool(self.config.basic.pretrained))

        # ResNet-51-ext which extends the Resnet-51 with additional layers
        if self.config.basic.model == "resnet_51-ext":
            return resnet51_ext(pretrained=bool(self.config.basic.pretrained), dropout=self.config.training.dropout)

        raise Exception("Model {} does not exist".format(self.config.basic.model))

    def _get_optimizer(self):
        """
        Get different optimizers for different experiments

        Returns
        -------
        An Optimizer
        """

        # SGD Optimizer
        if self.config.training.optimizer.type == "SGD":
            return optim.SGD(self.model_device.parameters(),
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

        Returns
        -------
        Loss Function
        """

        # Binary Cross Entropy with Logits Loss
        if self.config.training.criterion.type == "BCEWithLogitsLoss":
            return nn.BCEWithLogitsLoss()

        # Binary Cross Entropy Loss
        if self.config.training.criterion.type == "BCELoss":
            return nn.BCELoss()

        raise Exception("Criterion {} does not exist".format(self.config.training.criterion.type))
