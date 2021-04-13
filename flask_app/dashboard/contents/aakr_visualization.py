
import os
import numpy as np
import pandas as pd
import plotly.graph_objects as go

from core.aakr import AAKR
from core.data_handler import load_training_dataset, read_dataset

import dash_core_components as dcc
import dash_html_components as html
import dash_bootstrap_components as dbc
from dash.dependencies import Input, Output
from dash.exceptions import PreventUpdate


def make_plot(Y, Y_hat):
    """
    :param values: a numpy array
    """
    fig = go.Figure()
    x = np.arange(len(Y))
    fig.add_trace(
        go.Scatter(
            x=x,
            y=Y.to_numpy(),
            name="Obs.",
            mode="lines"
        )
    )
    fig.add_trace(
        go.Scatter(
            x=x,
            y=Y_hat.to_numpy(),
            name="Recon.",
            mode="lines"
        )
    )
    return dcc.Graph(figure=fig)

def create_layout(original_signal, reconstructed_signal):
    """
    Returns the plotly dash object containing the plots.

    Parameters
    ----------
    original_signal : pandas DataFrame
        It contains the original (corrupted) signal.

    reconstructed_signal : pandas DataFrame
        It contains the reconstructed signal.

    Return
    ------
    rows : list
        A list of rows containing the plots of the reconstructed signals.
    """
    columns = 2
    rows = []
    signals = sorted(list(set(original_signal.columns.get_level_values(0))))
    for signal in signals:
        rows.append(html.H4(signal))
        orig = original_signal[signal]
        recon = reconstructed_signal[signal]
        features = sorted(list(set(orig.columns.get_level_values(0))))
        col = []
        for i, feature in enumerate(features):
            fig = make_plot(
                Y=orig[feature],
                Y_hat=recon[feature]
            )
            card = dbc.Card([
                dbc.CardHeader(html.H5(feature)),
                dbc.CardBody(fig)
            ])
            col.append(
                html.Div(card, className="col")
            )
            if (i + 1) % columns == 0 and (i + 1) != len(features):
                rows.append(html.Div(col, className="row"))
                rows.append(html.Br())
                col = []
        rows.append(html.Div(col, className="row"))
        rows.append(html.Br())
    return rows

form_file = dbc.FormGroup(
    [
        dbc.Label(
            "Select the signal to reconstruct",
            html_for="dropdown-file",
        ),
        dcc.Dropdown(
            id="dropdown-file-aakr-test",
            multi=False,
            options=[{"label": i, "value": i} for i in
                     sorted(os.listdir("training_validation_2"))]
        )
    ]
)

form_percent = dbc.FormGroup([
    dbc.Label(
        "Select the amount of data in the training set out of all the " + \
        "available training data",
        html_for="aakr-amount-data"
    ),
    dbc.Input(
        id="aakr-amount-data",
        placeholder="% training data",
        type="number",
        min=25,
        max=100,
    )
])

aakr_viz = [
    html.Div(
        children=[
            html.Br(),
            html.Div(html.Div(form_percent, className="col"), className="row"),
            html.Br(),
            html.Div(html.Div(form_file, className="col"), className="row"),
            html.Br(),
            dcc.Loading(
                html.Div(
                    id="div-visualize-reconstruction",
                    className="accordion"
                )
            ),
        ],
        className = "container"
    )
]

def aakr_viz_callbacks(app):

    @app.callback(
        Output("div-visualize-reconstruction", "children"),
        [Input("aakr-amount-data", "value"),
         Input("dropdown-file-aakr-test", "value")]
    )
    def visulize_reconstructed_signal(percent_data, fname):
        if percent_data is None or fname is None:
            raise PreventUpdate
        X = load_training_dataset(percent_data=percent_data/100)
        Y = read_dataset(os.path.join("training_validation_2", fname))
        aakr = AAKR()
        Y, Y_hat = aakr.predict(X, Y)
        feature_names = aakr.features
        layout = create_layout(
            original_signal=pd.DataFrame(Y, columns=feature_names),
            reconstructed_signal=pd.DataFrame(Y_hat, columns=feature_names)
        )
        return layout
