#!/usr/bin/python3
"""
main evaluation file
"""
import random

from evaluation.evaluate_model import EvalModel
from utils.config_utils import get_config
from utils.utils import init_environment


def main(generate_accuracy_csv=False):
    """
    run evaluation for a specific model and generate chart
    """
    # Load configuration for evaluation
    # config = get_config('eval/resnet152')
    config = get_config('eval/affact')
    # config = get_config('eval/resnet51_s')

    # Init environment, use GPU if available, set random seed
    device = init_environment(config)

    # Create an evaluation instance with the loaded configuration on the loaded device
    eval_instance = EvalModel(config, device)

    # Calculate accuracies on for prediction of model
    # TODO: Check if csv not available.. then generate anyway
    if generate_accuracy_csv:

        # Run the evaluation
        df = eval_instance.evaluate()

        # Save csv
        df.to_csv('{}/evaluation_result.csv'.format(config.experiments_dir))

    # execute quantitative analysis
    if config.evaluation.quantitative.enabled:
        eval_instance.quantitative_analysis()

    # execute qualitative analysis
    if config.evaluation.qualitative.enabled:
        eval_instance.qualitative_analysis()



if __name__ == '__main__':
    main(generate_accuracy_csv=False)
