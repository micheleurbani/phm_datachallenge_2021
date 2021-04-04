
import numpy as np
import pandas as pd
from os import listdir
from itertools import product
import plotly.graph_objects as go
from core.data_handler import read_dataset

import dash_core_components as dcc
import dash_html_components as html
import dash_bootstrap_components as dbc
from dash.dependencies import Input, Output
from dash.exceptions import PreventUpdate


def make_card(values):
    """
    :param values: a numpy array
    """
    fig = go.Figure(
        data=go.Scatter(
            x=np.arange(len(values)),
            y=values
        )
    )
    fig.update_layout(
        title={
            "text": values.name[0] + " : " + values.name[1],
        }
    )
    return dbc.Card(
        dbc.CardBody(
            dcc.Graph(figure=fig)
        )
    )

form_folder = dbc.FormGroup(
    [
        dbc.Label(
            "Select folder",
            html_for="dropdown-folder",
        ),
        dcc.Dropdown(
            id="dropdown-folder",
            multi=False,
            value="training_validation_1",
            options=[
                {"label": "training_validation_1",
                "value": "training_validation_1"},
                {"label": "training_validation_2",
                "value": "training_validation_2"},
            ],
        )
    ]
)

form_file = dbc.FormGroup(
    [
        dbc.Label(
            "Select dataset",
            html_for="dropdown-file",
        ),
        dcc.Dropdown(
            id="dropdown-file",
            multi=False,
        )
    ]
)

signal_dropdown = dbc.FormGroup(
    [
        dbc.Label(
            "Select signal",
            html_for="dropdown-signal",
        ),
        dcc.Dropdown(
            id="dropdown-signal",
            multi=False,
        )
    ]
)

feature_dropdown = dbc.FormGroup(
    [
        dbc.Label(
            "Select feature",
            html_for="dropdown-feature",
        ),
        dcc.Dropdown(
            id="dropdown-feature",
            multi=False,
        )
    ]
)

select_dataset = html.Div(
    id="row-select-dataset",
    children=[
        html.Br(),
        html.Div(form_folder, className="col"),
        html.Div(form_file, className="col"),
    ],
    className="row"
)

select_signal = html.Div(
    id="row-select-signal",
    children=[
        html.Br(),
        html.Div(signal_dropdown, className="col"),
        html.Div(feature_dropdown, className="col"),
    ],
    className="row"
)

view_dataset = html.Div(
    id="dataset-visualization",
)

overview = html.Div(
    children=[
        html.Br(),
        html.Div(select_dataset, className="container"),
        html.Br(),
        html.Div(select_signal, className="container"),
        html.Br(),
        html.Div(view_dataset, className="container"),
    ]
)

def navigate_dataset_callbacks(app):

    @app.callback(
        Output("dropdown-file", "options"),
        Input("dropdown-folder", "value")
    )
    def update_dropdown_file_options(value):
        return [{"label": i, "value": i} for i in sorted(listdir(value))]

    @app.callback(
        Output("dropdown-signal", "options"),
        [Input("dropdown-folder", "value"),
        Input("dropdown-file", "value")]
    )
    def update_signal_dropdown(folder, fname):
        if folder is None or fname is None:
            raise  PreventUpdate
        # Load the dataset
        df = read_dataset(folder + "/" + fname)
        return [{"label": i, "value": i} for i in
                sorted(list(set(df.columns.get_level_values(0))))]

    @app.callback(
        Output("dropdown-feature", "options"),
        [Input("dropdown-folder", "value"),
        Input("dropdown-file", "value"),
        Input("dropdown-signal", "value")]
    )
    def update_feature_dropdown(folder, fname, signal):
        if folder is None or fname is None or signal is None:
            raise  PreventUpdate
        # Load the dataset
        df = read_dataset(folder + "/" + fname)
        return [{"label": i, "value": i} for i in
                sorted(list(set(df[signal].columns.get_level_values(0))))]

    @app.callback(
        Output("dataset-visualization", "children"),
        [Input("dropdown-folder", "value"),
         Input("dropdown-file", "value"),
         Input("dropdown-signal", "value"),
         Input("dropdown-feature", "value")]
    )
    def update_dataset_visualization(folder, fname, signal, feature):
        if folder is None or fname is None or signal is None or feature is \
            None:
            raise  PreventUpdate
        # Load the dataset
        df = read_dataset(folder + "/" + fname)
        return make_card(df[(signal, feature)])
