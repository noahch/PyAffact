#!/usr/bin/python3
"""
main evaluation file
"""

from evaluation.evaluate_model import EvalModel
from utils.config_utils import get_config
from utils.utils import init_environment
import os.path


def main(generate_accuracy_csv=False, order=None):
    """
    run evaluation for a specific model and generate chart
    """
    # Load configuration for evaluation
    # config = get_config('eval/resnet152')
    # config = get_config('eval/affact')
    config = get_config('eval/resnet51_s')

    # Init environment, use GPU if available, set random seed
    device = init_environment(config)

    # Create an evaluation instance with the loaded configuration on the loaded device
    eval_instance = EvalModel(config, device)

    # Check if flag is set to generate file that contains accuracies of the model of different test sets
    # Further check if such a file exists. If not, generate file regardless of the state of the flag.
    if generate_accuracy_csv or not os.path.isfile('{}/evaluation_result.csv'.format(config.experiments_dir)):
        # Run the evaluation (Calculate prediction accuracies of model on different test sets)
        df = eval_instance.evaluate()
        # Save csv
        df.to_csv('{}/evaluation_result.csv'.format(config.experiments_dir))

    # execute quantitative analysis
    if config.evaluation.quantitative.enabled:
        eval_instance.quantitative_analysis(order)

    # execute qualitative analysis
    if config.evaluation.qualitative.enabled:
        eval_instance.qualitative_analysis()



if __name__ == '__main__':
    # Define the order of your testsets. Set to None if order from accuracy file should be applied.
    order = ['testsetA', 'testsetC', 'testsetD', 'testsetT']
    # Overwrite if file containing accuracies should be generated. Only necessary the first time running the analysis.
    main(generate_accuracy_csv=False, order=order)
