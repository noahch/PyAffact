"""
Configuration Utils Function Collection
"""
import json
import os
import argparse
import collections
from datetime import datetime

import yaml
from dotmap import DotMap


def get_config(config_name=None):
    """
    Reads the configuration and returns a configuration DotMap

    Parameters
    ----------
    config_name Optional name of the configuation file

    Returns
    -------
    Configuration DotMap with arguments parsed form terminal/cmd
    """

    # Read arguments
    args = _read_arguments()

    # Read the config name specified
    if not config_name:
        config_name = args.__dict__['config.name']

    # load the file and parse it to a DotMap
    with open("config/" + config_name + ".yml", "r") as file:
        config_dict = yaml.safe_load(file)
    config = DotMap(config_dict)

    # Overwrite default values
    return _overwrite_defaults(config, args)


def _overwrite_defaults(config, args):
    """
    Overwrite the default values in the configuration DotMap

    Parameters
    ----------
    config configuration file
    args command line arguments

    Returns
    -------
    DotMap Configuration with new values
    """

    # Overwrite all arguments that are set via terminal/cmd
    for argument, argument_value in args.__dict__.items():
        if argument_value is not None:
            config = _replace_value_in_config(config, argument, argument_value)
    return config


def _read_arguments():
    """
    Read the arguments from the command line/terminal

    Returns
    -------
    ArgParser
    """
    parser = argparse.ArgumentParser(description='Arguments for PyAffact')
    parser.add_argument('--config.name', default='train/basic_config', type=str)
    parser.add_argument('--basic.cuda_device_name', default=None, type=str)
    parser.add_argument('--basic.experiment_name', default=None, type=str)
    parser.add_argument(
        '--basic.experiment_description',
        default=None,
        type=str)
    parser.add_argument('--basic.result_directory', default=None, type=str)
    parser.add_argument(
        '--basic.enable_wand_reporting',
        default=None,
        type=int)

    parser.add_argument('--model.name', default=None, type=str)
    parser.add_argument('--model.pretrained', default=None, type=int)
    parser.add_argument('--model.dropout', default=None, type=float)

    parser.add_argument(
        '--dataset.partition_filename',
        default=None,
        type=str)
    parser.add_argument(
        '--dataset.dataset_labels_filename',
        default=None,
        type=str)
    parser.add_argument(
        '--dataset.dataset_image_folder',
        default=None,
        type=str)
    parser.add_argument(
        '--dataset.landmarks_filename',
        default=None,
        type=str)
    parser.add_argument(
        '--dataset.bounding_boxes_filename',
        default=None,
        type=str)
    parser.add_argument(
        '--dataset.bounding_box_mode',
        default=None,
        type=int)

    parser.add_argument('--dataset.bounding_box_scale', default=None, type=int)
    parser.add_argument('--training.epochs', default=None, type=int)
    parser.add_argument('--training.save_frequency', default=None, type=int)
    parser.add_argument('--training.optimizer.type', default=None, type=str)
    parser.add_argument(
        '--training.optimizer.learning_rate',
        default=None,
        type=float)
    parser.add_argument(
        '--training.optimizer.momentum',
        default=None,
        type=float)
    parser.add_argument('--training.criterion.type', default=None, type=str)
    parser.add_argument('--training.lr_scheduler.type', default=None, type=str)
    parser.add_argument(
        '--training.lr_scheduler.step_size',
        default=None,
        type=int)
    parser.add_argument(
        '--training.lr_scheduler.gamma',
        default=None,
        type=float)
    parser.add_argument(
        '--training.lr_scheduler.patience',
        default=None,
        type=float)



    parser.add_argument(
        '--preprocessing.dataloader.batch_size',
        default=None,
        type=int)
    parser.add_argument(
        '--preprocessing.dataloader.shuffle',
        default=None,
        type=str)
    parser.add_argument(
        '--preprocessing.dataloader.num_workers',
        default=None,
        type=int)
    parser.add_argument(
        '--preprocessing.dataloader.prefetch_factor',
        default=None,
        type=int)

    parser.add_argument(
        '--preprocessing.save_preprocessed_image.enabled',
        default=None,
        type=int)
    parser.add_argument(
        '--preprocessing.save_preprocessed_image.frequency',
        default=None,
        type=int)
    parser.add_argument(
        '--preprocessing.transformation.crop_size.x',
        default=None,
        type=int)
    parser.add_argument(
        '--preprocessing.transformation.crop_size.y',
        default=None,
        type=int)
    parser.add_argument(
        '--preprocessing.transformation.random_bounding_box.enabled',
        default=None,
        type=int)
    parser.add_argument(
        '--preprocessing.transformation.scale_jitter.enabled',
        default=None,
        type=int)
    parser.add_argument(
        '--preprocessing.transformation.scale_jitter.normal_distribution.mean',
        default=None,
        type=int)
    parser.add_argument(
        '--preprocessing.transformation.scale_jitter.normal_distribution.std',
        default=None,
        type=int)
    parser.add_argument(
        '--preprocessing.transformation.angle_jitter.enabled',
        default=None,
        type=int)
    parser.add_argument(
        '--preprocessing.transformation.angle_jitter.normal_distribution.mean',
        default=None,
        type=int)
    parser.add_argument(
        '--preprocessing.transformation.angle_jitter.normal_distribution.std',
        default=None,
        type=int)
    parser.add_argument(
        '--preprocessing.transformation.shift_jitter.enabled',
        default=None,
        type=int)
    parser.add_argument(
        '--preprocessing.transformation.shift_jitter.normal_distribution.mean',
        default=None,
        type=int)
    parser.add_argument(
        '--preprocessing.transformation.shift_jitter.normal_distribution.std',
        default=None,
        type=int)
    parser.add_argument(
        '--preprocessing.transformation.mirror.enabled',
        default=None,
        type=int)
    parser.add_argument(
        '--preprocessing.transformation.mirror.probability',
        default=None,
        type=int)
    parser.add_argument(
        '--preprocessing.transformation.gaussian_blur.enabled',
        default=None,
        type=int)
    parser.add_argument(
        '--preprocessing.transformation.gaussian_blur.normal_distribution.mean',
        default=None,
        type=int)
    parser.add_argument(
        '--preprocessing.transformation.gaussian_blur.normal_distribution.std',
        default=None,
        type=int)
    parser.add_argument(
        '--preprocessing.transformation.gamma.enabled',
        default=None,
        type=int)
    parser.add_argument(
        '--preprocessing.transformation.gamma.normal_distribution.mean',
        default=None,
        type=int)
    parser.add_argument(
        '--preprocessing.transformation.gamma.normal_distribution.std',
        default=None,
        type=int)
    parser.add_argument(
        '--preprocessing.transformation.positive_scale.enabled',
        default=None,
        type=int)
    parser.add_argument(
        '--preprocessing.transformation.positive_scale.min',
        default=None,
        type=int)
    parser.add_argument(
        '--preprocessing.transformation.positive_scale.max',
        default=None,
        type=int)
    parser.add_argument(
        '--preprocessing.transformation.temperature.enabled',
        default=None,
        type=int)

    args = parser.parse_args()
    return args


def _replace_value_in_config(config, argument, argument_value):
    """
    Replaces a value in the DotMap

    Parameters
    ----------
    config Configuration DotMap
    argument Argument to overwrite
    argument_value Argument value

    Returns
    -------
    new DotMap with new Values
    """

    # Recursive Help function which creates a nested dict
    def _create_nested_dict(key, value):
        value = {key[-1]: value}
        new_key_list = key[0:-1]
        if len(new_key_list) > 1:
            return _create_nested_dict(new_key_list, value)
        return {new_key_list[0]: value}

    # Recursive Help function which updates a value
    def _update(key, value):
        for k, val in value.items():
            if isinstance(val, collections.abc.Mapping):
                key[k] = _update(key.get(k, {}), val)
            else:
                key[k] = val
        return key

    argument_keys = argument.split('.')
    new_dict = _create_nested_dict(argument_keys, argument_value)
    return DotMap(_update(config.toDict(), new_dict))


def create_result_directory(config):
    """
    Creates the result directory and updates the configuration file

    Parameters
    ----------
    config configuration file

    Returns
    -------

    """

    # Get current time
    now = datetime.now()

    # Create name of the directory using a the current time as well as the
    # experiment name
    directory_name = '{}-{:02d}-{:02d}-{:02d}-{:02d}-{:02d}-{}'.format(
        now.year,
        now.month,
        now.day,
        now.hour,
        now.minute,
        now.second,
        config.basic.experiment_name)

    # Join directory path
    result_directory = os.path.join(
        config.basic.result_directory, directory_name)
    if not os.path.exists(result_directory):
        try:
            os.makedirs(result_directory)
            config.basic.result_directory = result_directory
            config.basic.result_directory_name = directory_name
            save_config_to_file(config)
        except BaseException as exc:
            raise Exception('directory could not be created') from exc
    else:
        raise Exception('directory already exists')


def save_config_to_file(config):
    """
    Save configuration to disk

    Parameters
    ----------
    config DotMap Configuration

    Returns
    -------

    """
    file = open(os.path.join(config.basic.result_directory, 'config.yml'), "w")
    file.write(yaml.dump(config.toDict()))
    file.close()
