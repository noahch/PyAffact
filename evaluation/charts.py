import plotly.graph_objects as go


def generate_attribute_accuracy_plot(attribute_name, attribute_accuracy_train, attribute_accuracy_baseline_train,
                                     attribute_accuracy_val, attribute_accuracy_baseline_val):
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


def generate_attribute_accuracy_chart(labels, per_attribute_accuracy, per_attribute_baseline_accuracy,
                                      all_attributes_accuracy,
                                      all_attributes_baseline_accuracy):
    import plotly.graph_objects as go

    color_array = [
        'forestgreen' if round(x, 3) > round(per_attribute_baseline_accuracy[i], 3) else 'darkorange' if round(x,
                                                                                                               3) == round(
            per_attribute_baseline_accuracy[i], 3) else 'indianred' for i, x in enumerate(per_attribute_accuracy)]

    results = {
        'positive_val': [],
        'positive_label': [],
        'neutral_val': [],
        'neutral_label': [],
        'negative_val': [],
        'negative_label': [],
    }

    for i, x in enumerate(per_attribute_accuracy):
        if round(x, 3) > round(per_attribute_baseline_accuracy[i], 3):
            results['positive_val'].append(x)
            results['positive_label'].append(labels[i])
        elif round(x, 3) == round(per_attribute_baseline_accuracy[i], 3):
            results['neutral_val'].append(x)
            results['neutral_label'].append(labels[i])
        else:
            results['negative_val'].append(x)
            results['negative_label'].append(labels[i])

    if round(all_attributes_accuracy, 3) > round(all_attributes_baseline_accuracy, 3):
        results['positive_val'].append(all_attributes_accuracy)
        results['positive_label'].append('<b>OVERALL</b>')
    elif round(all_attributes_accuracy, 3) == round(all_attributes_baseline_accuracy, 3):
        results['neutral_val'].append(all_attributes_accuracy)
        results['neutral_label'].append('<b>OVERALL</b>')
    else:
        results['negative_val'].append(all_attributes_accuracy)
        results['negative_label'].append('<b>OVERALL</b>')

    labels.append('<b>OVERALL</b>')
    per_attribute_baseline_accuracy.append(all_attributes_baseline_accuracy)

    layout = go.Layout(
        autosize=False,
        margin=go.layout.Margin(
            l=2,
            r=2,
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
        y=labels,
        x=per_attribute_baseline_accuracy,
        name='Baseline Accuracy',
        marker_color='gray',
        orientation='h',
        offset=-0.40,
        marker_line_width=0
    ))
    fig.add_trace(go.Bar(
        y=results['positive_label'],
        x=results['positive_val'],
        name='Model Accuracy +',
        marker_color='#306e41',
        orientation='h',
        offset=0,
        marker_line_width=0
    ))
    fig.add_trace(go.Bar(
        y=results['neutral_label'],
        x=results['neutral_val'],
        name='Model Accuracy +/-',
        marker_color='#c96f2a',
        orientation='h',
        offset=0,
        marker_line_width=0
    ))
    fig.add_trace(go.Bar(
        y=results['negative_label'],
        x=results['negative_val'],
        name='Model Accuracy -',
        marker_color='#963034',
        orientation='h',
        offset=0,
        marker_line_width=0
    ))

    # Here we modify the tickangle of the xaxis, resulting in rotated labels.
    fig.update_layout(barmode='overlay', xaxis_tickangle=-45, bargroupgap=0.1, bargap=0.6)
    return fig


def accuracy_table(column_names, prediction, per_attribute_correct_classification):
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
        # paper_bgcolor='rgba(255,255,255,1)',
        # plot_bgcolor='rgba(255,255,255,0.5)'
    )

    header = ['<b>Attribute</b>'] + ['Image-{}'.format(x) for x in range(1, len(prediction) + 1)]
    cellText = [column_names] + [['yes' if y else 'no' for y in x] for x in prediction]
    colorlist = [['#306e41' if y else '#963034' for y in x] for x in per_attribute_correct_classification]
    colors = [['grey'] * len(column_names)] + colorlist
    table = go.Table(columnwidth=[70] + [30] * len(prediction),
                     header=dict(values=header),
                     cells=dict(values=cellText, fill_color=colors, font=dict(color='white'), align=['left'] + ['center'] * len(prediction))
                     )

    fig = go.Figure(layout=layout, data=table)
    return fig
