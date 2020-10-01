from config.config_utils import get_config, create_result_directory
from evaluation.evaluate_model import EvalModel
from training.train_model import TrainModel
from utils.utils import init_environment, get_gpu_memory_map
import sys
import argparse

# Init environment, use GPU if available, set random seed
device = init_environment()

get_gpu_memory_map()

# Load configuration for training
config = get_config()


if config.basic.mode == 'train':
    create_result_directory(config)
    # Create a training instance with the loaded configuration on the loaded device
    training_instance = TrainModel(config, device)

    # Run the training
    training_instance.train()

elif config.basic.mode == 'eval':
    # Create an evaluation instance with the loaded configuration on the loaded device
    eval_instance = EvalModel(config, device)

    # Run the training
    eval_instance.eval()
else:
    raise Exception('Mode does not exist')

