
import numpy as np
import plotly.graph_objects as go
from core.data_handler import load_training_dataset, read_dataset

import dash_core_components as dcc
import dash_html_components as html
import dash_bootstrap_components as dbc
from dash.dependencies import Input, Output
from dash.exceptions import PreventUpdate


def feature_plot(values):
    fig = go.Figure(
        data=go.Histogram(
            x=values
        )
    )
    fig.update_layout(
        title={
            "text": values.name,
        }
    )
    return fig

form_percent = dbc.FormGroup([
    dbc.Label(
        "Select the amount of data for visualization",
        html_for="amount-data"
    ),
    dbc.Input(
        id="amount-data",
        placeholder="% training data",
        type="number",
        min=25,
        max=100,
    )
])

dataset_parameter = html.Div(
    id="row-training-set-parameters",
    children=[
        html.Br(),
        html.Div(form_percent, className="col"),
    ],
    className="row",
)

signal_dropdown = dbc.FormGroup(
    [
        dbc.Label(
            "Select signal",
            html_for="dropdown-signal-training",
        ),
        dcc.Dropdown(
            id="dropdown-signal-training",
            multi=False,
        )
    ]
)

select_signal = html.Div(
    id="row-select-signal-training",
    children=[
        html.Br(),
        html.Div(signal_dropdown, className="col"),
    ],
    className="row"
)

overview_training = html.Div(
    children=[
        html.Br(),
        html.Div(dataset_parameter, className="container"),
        html.Br(),
        html.Div(select_signal, className="container"),
        html.Br(),
        dcc.Loading(
            children=html.Div(id="signal-visualization", className="container")
        )
    ]
)

def training_set_callbacks(app):

    @app.callback(
        Output("dropdown-signal-training", "options"),
        Input("amount-data", "value")
    )
    def update_signal_options(value):
        if value is None:
            raise PreventUpdate
        df = read_dataset("training_validation_1/class_0_0_data.csv")
        return [{"label": i, "value": i} for i in
                sorted(list(set(df.columns.get_level_values(0))))]

    @app.callback(
        Output("signal-visualization", "children"),
        [Input("dropdown-signal-training", "value"),
         Input("amount-data", "value")]
    )
    def update_visualization(signal, percent_data):
        if signal is None:
            raise PreventUpdate
        df = load_training_dataset(percent_data=percent_data/100)
        # Filter dataframe and hold only signal values
        df = df[signal]
        features = sorted(list(set(df.columns.get_level_values(0))))
        plots = []
        for feature in features:
            plots.append(
                html.Div(
                    dcc.Graph(figure=feature_plot(df[feature])),
                    className="row"
                )
            )
        return plots
