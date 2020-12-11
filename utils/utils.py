import random

import torch
import numpy as np
import logging
import subprocess

def init_environment(config):
    # CUDA for PyTorch
    use_cuda = torch.cuda.is_available()
    if len(config.basic.cuda_device_name.split(',')) > torch.cuda.device_count():
        raise Exception("Not enough devices")

    device = torch.device(config.basic.cuda_device_name.split(',')[0] if use_cuda else "cpu")
    if use_cuda:
        torch.cuda.empty_cache()
    # torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    # TODO: SEED AS PARAM
    np.random.seed(0)
    torch.cuda.manual_seed_all(0)
    torch.manual_seed(0)
    random.seed(0)
    setup_logging()
    return device

def setup_logging():
    logging.basicConfig(format='%(asctime)s.%(msecs)03d %(pathname)s:%(lineno)d %(levelname)s %(message)s', datefmt='%Y-%m-%d %H:%M:%S', level=logging.INFO)

def get_gpu_memory_map(id_string=''):
    """Get the current gpu usage.

    Returns
    -------
    usage: dict
        Keys are device ids as integers.
        Values are memory usage as integers in MB.
    """
    result = subprocess.check_output(
        [
            'nvidia-smi', '--query-gpu=memory.used',
            '--format=csv,nounits,noheader'
        ], encoding='utf-8')
    # Convert lines into a dictionary
    gpu_memory = [int(x) for x in result.strip().split('\n')]
    gpu_memory_map = dict(zip(range(len(gpu_memory)), gpu_memory))
    # print('{}:{}'.format(id_string, gpu_memory_map))
    return gpu_memory_map
