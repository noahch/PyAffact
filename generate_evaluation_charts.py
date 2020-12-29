"""
Generate chart to compare relative performance of two models
"""
from preprocessing.dataset_generator import generate_test_dataset
from utils.config_utils import get_config
import pandas as pd
import numpy as np
import plotly.graph_objects as go

def generate_relative_improvement_chart(labels, accuracy_dataframe_left,  accuracy_dataframe_right, left_name, right_name):
    labels.append('<b>OVERALL</b>')
    figures = []
    for test_set in accuracy_dataframe_left.columns:
        left_list = accuracy_dataframe_left[test_set].tolist()
        right_list = accuracy_dataframe_right[test_set].tolist()
        left_list.append(np.mean(left_list))
        right_list.append(np.mean(right_list))
        left_trace = []
        right_trace = []

        for i in range(len(left_list)):
            rel_change = ((right_list[i] - left_list[i]) / left_list[i]) * 100
            if rel_change > 10:
                print(i)
                print(test_set)
                print(right_list[i])
                print(left_list[i])
            if rel_change > 0:
                right_trace.append(rel_change)
                left_trace.append(0)
            else:
                left_trace.append(rel_change)
                right_trace.append(0)


        layout = go.Layout(
            autosize=False,
            margin=go.layout.Margin(
                l=2,
                r=3,
                b=2,
                t=5,
                pad=2
            ),
            height=750,
            width=550,
            # paper_bgcolor='rgba(255,255,255,1)',
            # plot_bgcolor='rgba(255,255,255,0.5)'
        )

        fig = go.Figure(layout=layout)
        fig.add_trace(go.Bar(
            y=labels[::-1],
            x=left_trace[::-1],
            name=left_name,
            orientation='h'
        ))
        fig.add_trace(go.Bar(
            y=labels[::-1],
            x=right_trace[::-1],
            name=right_name,
            orientation='h'
        ))
        fig.update_layout(xaxis=dict(
            range = [-15, 15],
            tickmode = 'array',
            tickvals = [-15, -10, -5, 0, 5, 10, 15],
            ticktext = ['-15%', '-10%', '-5%', '0%', '5%', '10%', '15%']
        ))
        figures.append(fig)
    return figures




if __name__ == '__main__':

    config_l = get_config('eval/2020-12-08-16-37-43-ResNet51')
    config_r = get_config('eval/2020-12-08-16-37-10-affact')
    left_name = 'ResNet-51'
    right_name = 'AFFACT'

    labels, _, _ = generate_test_dataset(config_l)
    labels = labels.columns.tolist()
    accuracy_df_l = pd.read_csv('{}/evaluation_result.csv'.format(config_l.experiments_dir),
                              index_col=0)
    accuracy_df_r = pd.read_csv('{}/evaluation_result.csv'.format(config_r.experiments_dir),
                              index_col=0)

    figures = generate_relative_improvement_chart(labels, accuracy_df_l, accuracy_df_r, 'ResNet-51', 'AFFACT')
    for i, test_set in enumerate(accuracy_df_l.columns):
        figures[i].show()
        figures[i].write_image('{}/eval_{}_{}_{}_relative_improvement.png'.format(config_l.experiments_dir, left_name , right_name, test_set), format='png', scale=3)
        figures[i].write_image('{}/eval_{}_{}_{}_relative_improvement.png'.format(config_r.experiments_dir, right_name, left_name,  test_set), format='png', scale=3)
