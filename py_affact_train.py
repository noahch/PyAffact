#!/usr/bin/python3
"""
Main File to start training the AFFACT Network
"""
from utils.config_utils import get_config, create_result_directory
from training.train_model import TrainModel
from utils.utils import init_environment


def main():
    """
    Run training for a specific model
    Returns
    -------

    """
    # Load configuration for training
    config = get_config()
    # Init environment, use GPU if available, set random seed
    device = init_environment(config)
    # Create result directory
    create_result_directory(config)
    # Create a training instance with the loaded configuration on the loaded device
    training_instance = TrainModel(config, device)
    # Run the training
    training_instance.train()


if __name__ == '__main__':
    main()
