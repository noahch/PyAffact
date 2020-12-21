#!/usr/bin/python3
"""
main evaluation file
"""
from evaluation.charts import generate_model_accuracy_of_testsets_2
from generate_evaluation_charts import _get_attribute_baseline_accuracy_val
from preprocessing.dataset_generator import generate_test_dataset
from utils.config_utils import get_config
from evaluation.evaluate_model import EvalModel
from utils.utils import init_environment
import torch
import pandas as pd


def main(generate_accuracy_csv=False):
    """
    run evaluation for a specific model and generate chart

    Returns
    -------

    """
    # Load configuration for training
    config = get_config('eval/resnet-51-affact-positive-scale_eval_config')

    # Calculate accuracies on for prediction of model
    if generate_accuracy_csv:

        # Init environment, use GPU if available, set random seed
        device = init_environment(config)

        # Create an evaluation instance with the loaded configuration on the loaded device
        eval_instance = EvalModel(config, device)

        # Run the evaluation
        df = eval_instance.eval()

        # Save csv
        df.to_csv('{}/evaluation_result.csv'.format(config.experiments_dir))


    # Load true labels
    labels, _, _ = generate_test_dataset(config)

    # Load accuracy DF from disk
    accuracy_df = pd.read_csv('{}/evaluation_result.csv'.format(config.experiments_dir),
                              index_col=0)

    # Calculate baseline Accuracies
    test_attribute_baseline_accuracy = _get_attribute_baseline_accuracy_val(labels, config)
    all_attributes_baseline_accuracy = test_attribute_baseline_accuracy.sum(axis=0) / labels.shape[1]
    per_attribute_baseline_accuracy = test_attribute_baseline_accuracy

    # Generate figure with accuracies on different test sets for model and baseline
    figure = generate_model_accuracy_of_testsets_2(labels.columns.tolist(), accuracy_df,
                                                   per_attribute_baseline_accuracy.tolist(),
                                                   all_attributes_baseline_accuracy)
    # Save the figure
    figure.write_image('{}/eval_fig.png'.format(config.experiments_dir))


if __name__ == '__main__':
    main(generate_accuracy_csv=False)
