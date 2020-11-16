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
