#!/usr/bin/python3
from config.config_utils import get_config, create_result_directory
from evaluation.evaluate_model import EvalModel
from training.train_model import TrainModel
from utils.utils import init_environment, get_gpu_memory_map
import logging
import sys
import argparse

# Load configuration for training
config = get_config()

# Init environment, use GPU if available, set random seed
device = init_environment(config)

get_gpu_memory_map()

#TODO: assert mode = train, eval or train and eval


if config.basic.mode == 'train' or config.basic.mode == 'trainEval':
    create_result_directory(config)
    # Create a training instance with the loaded configuration on the loaded device
    training_instance = TrainModel(config, device)
    # logger = logging.getLogger('PyAffact')
    # logger.warning('test')
    logging.warning('test')
    # Run the training
    training_instance.train()

if config.basic.mode == 'eval' or config.basic.mode == 'trainEval':
    config.basic.mode = 'eval'
    config.transformation.scale_jitter.enabled = 0
    config.transformation.angle_jitter.enabled = 0
    config.transformation.shift_jitter.enabled = 0
    config.transformation.mirror.enabled = 0
    config.transformation.gaussian_blur.enabled = 0
    config.transformation.gamma.enabled = 0
    # Create an evaluation instance with the loaded configuration on the loaded device
    eval_instance = EvalModel(config, device)

    # Run the training
    eval_instance.eval()

