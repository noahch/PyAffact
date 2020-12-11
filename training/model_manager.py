from torch import optim, nn
from torch.optim import lr_scheduler

from network.resnet_51 import resnet50
from network.resnet_noahAndYves import resnet50_noahAndYves


class ModelManager():
    def __init__(self, config, device):
        self.config = config
        self.device = device
        self.model = nn.DataParallel(self._get_model(), device_ids=[6,7])
        self.model_device = self.model.to(self.device)

    def _get_model(self):
        if self.config.basic.model == "resnet_51":
            return resnet50(pretrained=bool(self.config.basic.pretrained))
        if self.config.basic.model == "resnet_noahAndYves":
            return resnet50_noahAndYves(pretrained=bool(self.config.basic.pretrained), dropout=self.config.training.dropout)
        raise Exception("Model {} does not exist".format(self.config.basic.model))

    def _get_optimizer(self):
        if self.config.training.optimizer.type == "SGD":
            return optim.SGD(self.model_device.parameters(),
                             lr=self.config.training.optimizer.learning_rate,
                             momentum=self.config.training.optimizer.momentum)
        if self.config.training.optimizer.type == "Adam":
            return optim.Adam(self.model_device.parameters(),
                             lr=self.config.training.optimizer.learning_rate)
        raise Exception("Optimizer {} does not exist".format(self.config.training.optimizer.type))

    def _get_criterion(self):
        if self.config.training.criterion.type == "BCEWithLogitsLoss":
            return nn.BCEWithLogitsLoss()
        if self.config.training.criterion.type == "BCELoss":
            return nn.BCELoss()
        raise Exception("Criterion {} does not exist".format(self.config.training.criterion.type))
