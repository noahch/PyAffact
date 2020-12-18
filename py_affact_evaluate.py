#!/usr/bin/python3
from utils.config_utils import get_config
from evaluation.evaluate_model import EvalModel
from utils.utils import init_environment
import torch


if __name__ == '__main__':
    torch.multiprocessing.set_start_method('spawn')
    # Load configuration for training
    config = get_config()

    # Init environment, use GPU if available, set random seed
    device = init_environment(config)

    # Create an evaluation instance with the loaded configuration on the loaded device
    eval_instance = EvalModel(config, device)

    # Run the training
    eval_instance.eval()
