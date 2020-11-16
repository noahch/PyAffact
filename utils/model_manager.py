from torch import optim, nn
from torch.optim import lr_scheduler

from network.auto_encoder import ConvAutoencoder
from network.auto_encoder_flat import FlatAutoencoder


class ModelManager():
    def __init__(self, config, device):
        self.config = config
        self.device = device
        self.model = self._get_model()
        self.model_device = self.model.to(self.device)

    def _get_model(self):
        if self.config.basic.model == "auto_encoder":
            return ConvAutoencoder()
        if self.config.basic.model == "auto_encoder_flat":
            return FlatAutoencoder(None)
        raise Exception("Model {} does not exist".format(self.config.basic.model))

    def _get_model_shared_layer(self, central):
        if self.config.basic.model == "auto_encoder":
            return ConvAutoencoder()
        if self.config.basic.model == "auto_encoder_flat":
            return FlatAutoencoder(central)
        raise Exception("Model {} does not exist".format(self.config.basic.model))
    def _get_optimizer(self):
        if self.config.training.optimizer.type == "SGD":
            return optim.SGD(self.model_device.parameters(),
                         lr=self.config.training.optimizer.learning_rate,
                         momentum=self.config.training.optimizer.momentum)
        if self.config.training.optimizer.type == "Adam":
            return optim.Adam(self.model_device.parameters(), lr=self.config.training.optimizer.learning_rate)
        raise Exception("Optimizer {} does not exist".format(self.config.training.optimizer.type))

    def _get_criterion(self):
        if self.config.training.criterion.type == "BCEWithLogitsLoss":
            return nn.BCEWithLogitsLoss()
        if self.config.training.criterion.type == "BCELoss":
            return nn.BCELoss()
        if self.config.training.criterion.type == "MSELoss":
            return nn.MSELoss()
        raise Exception("Criterion {} does not exist".format(self.config.training.criterion.type))
