"""
Generate chart to compare relative performance of two models
"""
from evaluation.charts import generate_relative_improvement_chart
from preprocessing.dataset_generator import generate_test_dataset
from utils.config_utils import get_config
import pandas as pd


if __name__ == '__main__':
    config_l = get_config('eval/resnet51_s')
    config_r = get_config('eval/affact')
    left_name = 'ResNet-51'
    right_name = 'AFFACT'

    labels, _, _ = generate_test_dataset(config_l)
    labels = labels.columns.tolist()
    accuracy_df_l = pd.read_csv('{}/evaluation_result.csv'.format(config_l.experiments_dir),
                              index_col=0)
    accuracy_df_r = pd.read_csv('{}/evaluation_result.csv'.format(config_r.experiments_dir),
                              index_col=0)

    figures = generate_relative_improvement_chart(labels, accuracy_df_l, accuracy_df_r, left_name,  right_name)
    for i, test_set in enumerate(accuracy_df_l.columns):
        figures[i].show()
        figures[i].write_image('{}/eval_{}_{}_{}_relative_improvement.png'.format(config_l.experiments_dir, left_name , right_name, test_set), format='png', scale=3)
        figures[i].write_image('{}/eval_{}_{}_{}_relative_improvement.png'.format(config_r.experiments_dir, left_name , right_name, test_set), format='png', scale=3)

