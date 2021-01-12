"""
functions that generate charts for evaluation
"""
import math
import string

import matplotlib
import plotly.graph_objects as go
import numpy as np
import matplotlib.pyplot as plt

from evaluation.utils import tensor_to_image


def generate_attribute_accuracy_plot(attribute_name, attribute_accuracy_train, attribute_accuracy_baseline_train,
                                     attribute_accuracy_val, attribute_accuracy_baseline_val):
    """
    Generates progress chart of an attribute during training and compares accuracy of training/validation data vs baseline model (majority guess from training data)
    :param attribute_name: name of attribute
    :param attribute_accuracy_train: accuracy on training data
    :param attribute_accuracy_baseline_train:  accuracy of majority guess on attribute (training)
    :param attribute_accuracy_val: accuracy on validation data
    :param attribute_accuracy_baseline_val: accuracy of majority guess on attribute (validation)
    :return: accuracy chart
    """
    layout = go.Layout(
        autosize=False,
        margin=go.layout.Margin(
            l=2,
            r=2,
            b=2,
            t=2,
            pad=2
        ),
        xaxis=go.layout.XAxis(
            title='Epochs',
            tickmode='array',
            tickvals=[x for x in range(1, len(attribute_accuracy_train) + 1)]
        ),
        yaxis=go.layout.YAxis(
            title='Accuracy',
            range=[0, 1],
            dtick=0.25,
            autorange=False,
            tickformat='.0%'
        )
    )

    fig = go.Figure(layout=layout)
    fig.add_trace(go.Scatter(
        x=[x for x in range(1, len(attribute_accuracy_train) + 1)],
        y=attribute_accuracy_train,
        name='Model Prediction Train ' + attribute_name
    ))
    fig.add_trace(go.Scatter(
        x=[x for x in range(1, len(attribute_accuracy_train) + 1)],
        y=[attribute_accuracy_baseline_train for x in range(1, len(attribute_accuracy_train) + 1)],
        name='Baseline Train ' + attribute_name
    ))
    fig.add_trace(go.Scatter(
        x=[x for x in range(1, len(attribute_accuracy_val) + 1)],
        y=attribute_accuracy_val,
        name='Model Prediction Val ' + attribute_name
    ))
    fig.add_trace(go.Scatter(
        x=[x for x in range(1, len(attribute_accuracy_val) + 1)],
        y=[attribute_accuracy_baseline_val for x in range(1, len(attribute_accuracy_val) + 1)],
        name='Baseline Val ' + attribute_name
    ))
    return fig



def generate_model_accuracy_of_testsets(labels, accuracy_dataframe, per_attribute_baseline_accuracy, all_attributes_baseline_accuracy, cut_off_threshold=15):
    """
    evaluates a model on different test sets
    :param labels: attribute names
    :param accuracy_dataframe: dataframe containing accuracy of all the test sets
    :param per_attribute_baseline_accuracy: majority guess accuracy per attribute
    :param all_attributes_baseline_accuracy: overall accuracy of majority guess
    :param cut_off_threshold: thresholds that defines when the bars are cut off and labels containing the real accuracy are displayed (in percentage)
    :return: evaluation chart
    """

    # data structures needed to create the chart
    labels.append('<b>OVERALL</b>')
    per_attribute_baseline_accuracy.append(all_attributes_baseline_accuracy)
    names = accuracy_dataframe.columns.tolist()
    table = str.maketrans('', '', string.ascii_lowercase)
    names_short = {x: x.translate(table) for x in names}
    colors = ['#4380b5', '#4aba8d', '#ba4aa5', '#b55c47', '#d4c557', '#a7cdd1']
    colors_cut_off = ['#4380b5', '#4aba8d', '#ba4aa5', '#b55c47', '#d4c557', '#a7cdd1']

    # use these colors if cut off bars should be darker
    # colors_cut_off = ['#335d82', '#338262', '#6e2b61', '#6e382b', '#827935', '#8bacb0']

    # initial layout
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

    )

    # create data structures needed to create the chart
    all_accuracies = dict()
    all_labels = dict()
    baseline_labels = []
    for i, testset in enumerate(accuracy_dataframe.columns.tolist()):
        accuracies = accuracy_dataframe[testset].tolist()
        accuracies.append(np.mean(accuracy_dataframe[testset].tolist()))
        all_accuracies[testset] = accuracies
        all_labels[testset] = []

    # loop that finds the longest bar chart per attribute group
    # if the longest bar chart exceeds the cut of threshold,
    # a custom label containing the real accuracies of all cut off bars in an attribute group is created
    for i in range(len(per_attribute_baseline_accuracy)):
        label = ''
        br_count = 0
        val = (1- per_attribute_baseline_accuracy[i])*100
        highest_idx = -1
        highest_val = min(val, cut_off_threshold)
        if min(val, cut_off_threshold) >= cut_off_threshold:
            label += 'MG:{:.2f}%'.format(val)
            br_count += 1
        for j in all_accuracies.keys():
            val2 = (1- all_accuracies[j][i])*100
            if min(val2, cut_off_threshold) > highest_val:
                highest_idx = j
                highest_val = min(val2, cut_off_threshold)
            if val2 > cut_off_threshold:
                if label == '':
                    label = '{}:{:.2f}%'.format(names_short[j], val2)
                    br_count += 1
                else:
                    if br_count % 3 == 0:
                        label += '<br>{}:{:.2f}%'.format(names_short[j], val2)
                    else:
                        label += ', {}:{:.2f}%'.format(names_short[j], val2)
                    br_count += 1
        if highest_idx == -1:
            baseline_labels.append(label)
            for k in all_accuracies.keys():
                all_labels[k].append('')
        else:
            baseline_labels.append('')
            for k in all_accuracies.keys():
                if k != highest_idx:
                    all_labels[k].append('')
                else:
                    all_labels[k].append(label)

    fig = go.Figure(layout=layout)

    # hack in order to have the right sorting of the bars
    i = 1
    for k in ['testsetA', 'testsetC', 'testsetD', 'testsetT'][::-1]:

        v = all_accuracies[k]
        fig.add_trace(go.Bar(
            y=labels[::-1],
            x=[min((1-x)*100, cut_off_threshold) for x in v][::-1],
            name=k,
            marker_color=[colors_cut_off[i] if ((1-x)*100)>cut_off_threshold else colors[i] for x in v][::-1],
            text=all_labels[k][::-1],
            textfont_size=6,
            textposition="outside",
            orientation='h'
        ))
        i = i + 1

    fig.add_trace(go.Bar(
        y=labels[::-1],
        x=[min((1-x)*100, cut_off_threshold) for x in per_attribute_baseline_accuracy][::-1],
        name='Majority Guess',
        marker_color=[colors_cut_off[0] if ((1-x)*100)>cut_off_threshold else colors[0] for x in per_attribute_baseline_accuracy][::-1],
        text=baseline_labels[::-1],
        textfont_size=6,
        textposition="outside",
        orientation='h'
    ))

    fig.update_layout(xaxis = dict(
        range = [0, 27],
        tickmode = 'array',
        tickvals = [x for x in range(0, cut_off_threshold+1, (cut_off_threshold//5))],
        ticktext = ['{:.2f}%'.format(x) for x in range(0, cut_off_threshold+1, (cut_off_threshold//5))]
    ),
        legend={'traceorder': 'reversed'},
        uniformtext=dict(minsize=6, mode='show'))
    return fig


def prediction_vs_ground_truth_chart(attribute_names, prediction, per_attribute_correct_classification):
    """
    generate chart that compares prediction vs ground truth on a selected sample
    :param attribute_names: attribute names
    :param prediction: list of predictions by the model
    :param per_attribute_correct_classification: list of ground truth labels
    :return: prediction vs ground truth chart
    """
    layout = go.Layout(
        autosize=False,
        margin=go.layout.Margin(
            l=2,
            r=2,
            b=2,
            t=5,
            pad=2
        ),
        height=1000

    )

    header = ['<b>Attribute</b>'] + ['Image-{}'.format(x) for x in range(1, len(prediction) + 1)]
    cell_text = [attribute_names] + [['yes' if y else 'no' for y in x] for x in prediction]
    color_list = [['#306e41' if y else '#963034' for y in x] for x in per_attribute_correct_classification]
    colors = [['grey'] * len(attribute_names)] + color_list
    table = go.Table(columnwidth=[70] + [30] * len(prediction),
                     header=dict(values=header),
                     cells=dict(values=cell_text, fill_color=colors, font=dict(color='white'), align=['left'] + ['center'] * len(prediction))
                     )

    fig = go.Figure(layout=layout, data=table)
    return fig


def accuracy_sample_plot(image_list, accuracy_list, number_of_img_per_row=3):
    """
    Generates a grid with images and their according overall accuracy
    :param image_list: list of images
    :param accuracy_list: list of accuracies
    :param number_of_img_per_row: number of images per row
    :return: 
    """
    matplotlib.use('Agg')
    plt.clf()
    plot_rows = math.ceil(len(image_list) / number_of_img_per_row)
    fig = plt.figure()
    fig.set_size_inches(5 * number_of_img_per_row, 5.5 * plot_rows)
    fig.tight_layout()
    for i in range(plot_rows):
        for j in range(number_of_img_per_row):
            current_index = ((i * number_of_img_per_row) + j)
            if current_index >= len(accuracy_list):
                break
            ax = plt.subplot2grid((plot_rows, number_of_img_per_row), (i, j))
            ax.set_title('Image-{}\nAccuracy: {}%'.format(current_index + 1, accuracy_list[current_index] * 100),
                         fontsize=20)
            ax.axis('off')
            ax.imshow(tensor_to_image(image_list[current_index]))

    return fig


def generate_relative_improvement_chart(labels, accuracy_dataframe_left,  accuracy_dataframe_right, left_name, right_name):
    """
    Generates chart comparing relative improvements of different models
    :param labels: labels/attributes
    :param accuracy_dataframe_left: model accuracies left
    :param accuracy_dataframe_right: model accuracies right
    :param left_name: name of left model
    :param right_name: name of right model
    :return: list of figures
    """
    labels.append('<b>OVERALL</b>')
    figures = []
    for test_set in accuracy_dataframe_left.columns:
        left_list = accuracy_dataframe_left[test_set].tolist()
        right_list = accuracy_dataframe_right[test_set].tolist()
        left_list.append(np.mean(left_list))
        right_list.append(np.mean(right_list))
        left_trace = []
        right_trace = []

        highest_val = 0
        for i in range(len(left_list)):
            a = (1 - left_list[i])
            b = (1 - right_list[i])
            switched = False
            if a > b:
                temp = a
                a = b
                b = temp
                switched = True


            rel_change = ((b - a) / a) * 100
            if abs(rel_change) > highest_val:
                highest_val = abs(rel_change)

            if switched:
                right_trace.append(rel_change)
                left_trace.append(0)
            else:
                left_trace.append(-rel_change)
                right_trace.append(0)


        layout = go.Layout(
            autosize=False,
            margin=go.layout.Margin(
                l=2,
                r=3,
                b=2,
                t=30,
                pad=2
            ),
            height=750,
            width=550,
            legend={'traceorder': 'reversed'},
            title='Relative Performance Differnece on {}. {} vs {}'.format(test_set, left_name, right_name),
            titlefont=dict(size=12)
            # paper_bgcolor='rgba(255,255,255,1)',
            # plot_bgcolor='rgba(255,255,255,0.5)'
        )

        fig = go.Figure(layout=layout)

        fig.add_trace(go.Bar(
            y=labels[::-1],
            x=right_trace[::-1],
            name=right_name,
            orientation='h'
        ))

        fig.add_trace(go.Bar(
            y=labels[::-1],
            x=left_trace[::-1],
            name=left_name,
            orientation='h'
        ))

        highest = int(math.ceil(highest_val / 10.0)) * 10
        tickvals = [-highest, -highest/2, 0, highest/2, highest]

        fig.update_layout(xaxis=dict(
            range = [-highest, highest],
            tickmode = 'array',
            tickvals = tickvals,
            ticktext = ['+{}%'.format(abs(x)) for x in tickvals]
        ))
        figures.append(fig)
    return figures