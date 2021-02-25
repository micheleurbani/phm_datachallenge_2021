
from os import listdir

import dash_core_components as dcc
import dash_html_components as html
import dash_bootstrap_components as dbc
from dash.dependencies import Input, Output


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

select_dataset = html.Div(
    id="row-select-dataset",
    children=[
        html.Br(),
        html.Div(form_folder, className="col"),
        html.Div(form_file, className="col"),
    ],
    className="row"
)

overview = html.Div(
    children=[
        html.Div(select_dataset, className="container"),
    ]
)

def navigate_dataset_callbacks(app):

    @app.callback(
        Output("dropdown-file", "options"),
        Input("dropdown-folder", "value")
    )
    def update_dropdown_file_options(value):
        return [{"label": i, "value": i} for i in sorted(listdir(value))]
