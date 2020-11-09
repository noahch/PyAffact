import plotly.graph_objects as go


def generate_attribute_accuracy_plot(attribute_name, attribute_accuracy, attribute_accuracy_baseline):
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
            tickvals=[x for x in range(1, len(attribute_accuracy)+1)]
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
        x=[x for x in range(1, len(attribute_accuracy)+1)],
        y=attribute_accuracy,
        name='Model Prediction ' + attribute_name
    ))
    fig.add_trace(go.Scatter(
        x=[x for x in range(1, len(attribute_accuracy)+1)],
        y=[attribute_accuracy_baseline for x in range(1, len(attribute_accuracy)+1)],
        name='Baseline ' + attribute_name
    ))
    return fig
