"""
Environment Utils File
"""
import os
import random
import logging
import shutil

import torch
import numpy as np


def init_environment(config):
    """
    Initialize the environment for training on GPUs

    Parameters
    ----------
    config Configuration DotMap

    Returns
    -------
    The cuda device
    """

    # CUDA for PyTorch
    use_cuda = torch.cuda.is_available()

    # Check if enough devices on current system
    if len(config.basic.cuda_device_name.split(',')) > torch.cuda.device_count():
        raise Exception("Not enough devices")

    # Specify device to use
    device = torch.device(config.basic.cuda_device_name.split(',')[0] if use_cuda else "cpu")

    # Empty cache to start on a cleared GPU
    if use_cuda:
        torch.cuda.empty_cache()

    # Set seeds
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(0)
    torch.cuda.manual_seed_all(0)
    torch.manual_seed(0)
    random.seed(0)

    # setup logging
    logging.basicConfig(
        format='%(asctime)s.%(msecs)03d %(pathname)s:%(lineno)d %(levelname)s %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S',
        level=logging.INFO)

    return device


def create_directory(directory, recreate=False):
    """
    Helper function to create a directory

    Parameters
    ----------
    directory Directory Path
    recreate If True, removes folder and creates new one

    Returns
    -------

    """
    if not os.path.exists(directory):
        try:
            os.makedirs(directory)
        except BaseException as exc:
            raise Exception('directory could not be created') from exc
    else:
        if recreate:
            shutil.rmtree(directory, ignore_errors=True)
            create_directory(directory)
        else:
            raise Exception('directory already exists')