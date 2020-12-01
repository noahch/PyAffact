import json
from dotmap import DotMap
import os
from pathlib import Path
from datetime import datetime
import argparse
import collections

def get_config():
    args = _read_arguments()
    with open("config/" + args.__dict__['config.name'] + ".json", "r") as f:
        my_dict = json.load(f)
    config = DotMap(my_dict)
    return _overwrite_defaults(config, args)

def _overwrite_defaults(config, args):
    for argument, argument_value in args.__dict__.items():
        if argument_value:
            config = _replace_value_in_config(config, argument, argument_value)
    return config

def _read_arguments():
    parser = argparse.ArgumentParser(description='Arguments for PyAffact')
    parser.add_argument('--config.name', default='basic_config', type=str)
    parser.add_argument('--basic.cuda_device_name', default=None, type=str)
    parser.add_argument('--basic.experiment_name', default=None, type=str)
    parser.add_argument('--basic.experiment_description', default=None, type=str)
    parser.add_argument('--basic.model', default=None, type=str)
    parser.add_argument('--basic.pretrained', default=None, type=int)
    parser.add_argument('--basic.result_directory', default=None, type=str)
    parser.add_argument('--basic.mode', default=None, type=str)
    parser.add_argument('--basic.enable_wand_reporting', default=None, type=int)

    parser.add_argument('--training.epochs', default=None, type=int)
    parser.add_argument('--training.save_frequency', default=None, type=int)
    parser.add_argument('--training.optimizer.type', default=None, type=str)
    parser.add_argument('--training.optimizer.learning_rate', default=None, type=float)
    parser.add_argument('--training.optimizer.momentum', default=None, type=float)
    parser.add_argument('--training.criterion.type', default=None, type=str)
    parser.add_argument('--training.lr_scheduler.type', default=None, type=str)
    parser.add_argument('--training.lr_scheduler.step_size', default=None, type=int)
    parser.add_argument('--training.lr_scheduler.gamma', default=None, type=float)

    parser.add_argument('--preprocessing.dataset.train_fraction', default=None, type=int)
    parser.add_argument('--preprocessing.dataset.val_fraction', default=None, type=int)
    parser.add_argument('--preprocessing.dataset.test_fraction', default=None, type=int)
    parser.add_argument('--preprocessing.dataset.number_of_samples', default=None, type=int)
    parser.add_argument('--preprocessing.dataset.dataset_labels_filename', default=None, type=str)
    parser.add_argument('--preprocessing.dataset.dataset_image_folder', default=None, type=str)
    parser.add_argument('--preprocessing.dataset.uses_landmarks', default=None, type=int)
    parser.add_argument('--preprocessing.dataset.uses_bounding_boxes', default=None, type=int)
    parser.add_argument('--preprocessing.dataset.landmarks_filename', default=None, type=str)
    parser.add_argument('--preprocessing.dataset.bounding_boxes_filename', default=None, type=str)

    parser.add_argument('--preprocessing.dataloader.batch_size', default=None, type=int)
    parser.add_argument('--preprocessing.dataloader.shuffle', default=None, type=str)
    parser.add_argument('--preprocessing.dataloader.num_workers', default=None, type=int)

    parser.add_argument('--preprocessing.transformation.use_affact_transformator', default=None, type=int)
    parser.add_argument('--preprocessing.transformation.save_transformation_image.enabled', default=None, type=int)
    parser.add_argument('--preprocessing.transformation.save_transformation_image.frequency', default=None, type=int)
    parser.add_argument('--preprocessing.transformation.crop_size.x', default=None, type=int)
    parser.add_argument('--preprocessing.transformation.crop_size.y', default=None, type=int)
    parser.add_argument('--preprocessing.transformation.random_bounding_box.enabled', default=None, type=int)
    parser.add_argument('--preprocessing.transformation.scale_jitter.enabled', default=None, type=int)
    parser.add_argument('--preprocessing.transformation.scale_jitter.normal_distribution.mean', default=None, type=int)
    parser.add_argument('--preprocessing.transformation.scale_jitter.normal_distribution.std', default=None, type=int)
    parser.add_argument('--preprocessing.transformation.angle_jitter.enabled', default=None, type=int)
    parser.add_argument('--preprocessing.transformation.angle_jitter.normal_distribution.mean', default=None, type=int)
    parser.add_argument('--preprocessing.transformation.angle_jitter.normal_distribution.std', default=None, type=int)
    parser.add_argument('--preprocessing.transformation.shift_jitter.enabled', default=None, type=int)
    parser.add_argument('--preprocessing.transformation.shift_jitter.normal_distribution.mean', default=None, type=int)
    parser.add_argument('--preprocessing.transformation.shift_jitter.normal_distribution.std', default=None, type=int)
    parser.add_argument('--preprocessing.transformation.mirror.enabled', default=None, type=int)
    parser.add_argument('--preprocessing.transformation.mirror.probability', default=None, type=int)
    parser.add_argument('--preprocessing.transformation.gaussian_blur.enabled', default=None, type=int)
    parser.add_argument('--preprocessing.transformation.gaussian_blur.normal_distribution.mean', default=None, type=int)
    parser.add_argument('--preprocessing.transformation.gaussian_blur.normal_distribution.std', default=None, type=int)
    parser.add_argument('--preprocessing.transformation.gamma.enabled', default=None, type=int)
    parser.add_argument('--preprocessing.transformation.gamma.normal_distribution.mean', default=None, type=int)
    parser.add_argument('--preprocessing.transformation.gamma.normal_distribution.std', default=None, type=int)

    parser.add_argument('--evaluation.test_labels_pickle_filename', default=None, type=str)
    parser.add_argument('--evaluation.test_landmarks_pickle_filename', default=None, type=str)
    parser.add_argument('--evaluation.train_majority_pickle_filename', default=None, type=str)
    parser.add_argument('--evaluation.model_weights_filename', default=None, type=str)
    parser.add_argument('--evaluation.qualitative.enabled', default=None, type=int)
    parser.add_argument('--evaluation.qualitative.number_of_images_per_row', default=None, type=int)
    parser.add_argument('--evaluation.quantitative.enabled', default=None, type=int)
    parser.add_argument('--evaluation.dataloader.batch_size', default=None, type=int)
    parser.add_argument('--evaluation.dataloader.shuffle', default=None, type=str)
    parser.add_argument('--evaluation.dataloader.num_workers', default=None, type=int)

    args = parser.parse_args()
    return args

def _update(d, u):
    for k, v in u.items():
        if isinstance(v, collections.abc.Mapping):
            d[k] = _update(d.get(k, {}), v)
        else:
            d[k] = v
    return d
def _create_nested_dict(k, v):
    value = {k[-1]: v}
    new_key_list = k[0:-1]
    if len(new_key_list) > 1:
        return _create_nested_dict(new_key_list, value)
    else:
        return {new_key_list[0]: value}

def _replace_value_in_config(config, argument, argument_value):
    argument_keys = argument.split('.')
    new_dict = _create_nested_dict(argument_keys, argument_value)
    return DotMap(_update(config.toDict(), new_dict))

def create_result_directory(config):
    # abs_path = Path(os.path.abspath(os.path.dirname(__file__)))
    # abs_path_parent = abs_path.parent
    now = datetime.now()
    directory_name = '{}-{:02d}-{:02d}-{:02d}-{:02d}-{:02d}-{}'.format(now.year, now.month, now.day, now.hour, now.minute, now.second, config.basic.experiment_name)
    result_directory = os.path.join(config.basic.result_directory, directory_name)
    if not os.path.exists(result_directory):
        try:
            os.makedirs(result_directory)
            config.basic.result_directory = result_directory
            config.basic.result_directory_name = directory_name
            save_config_to_file(config)
            return config
        except:
            raise Exception('directory could not be created')
    else:
        raise Exception('directory already exists')

def save_config_to_file(config):
    f = open(os.path.join(config.basic.result_directory, 'config.json'), "w")
    f.write(json.dumps(config.toDict(), indent=2))
    f.close()