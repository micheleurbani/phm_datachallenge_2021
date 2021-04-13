from dash import Dash
import dash_html_components as html
import dash_bootstrap_components as dbc

from .layout import html_layout
from .contents.navigate_dataset import overview, navigate_dataset_callbacks
from .contents.explore_training_set import (
    overview_training,
    training_set_callbacks,
)
from .contents.aakr_visualization import aakr_viz, aakr_viz_callbacks


def init_dashboard(server):
    """Create a Plotly Dash dashboard."""
    dash_app = Dash(
        server=server,
        routes_pathname_prefix='/dashapp/',
        external_stylesheets=[
            'https://cdn.jsdelivr.net/npm/bootstrap@5.0.0-beta1/' +
            'dist/css/bootstrap.min.css',
        ]
    )

    dash_app.index_string = html_layout
    # Create Dash Layout
    card_tabs = dbc.Tabs(
        id="homepage-tabs",
        children=[
            dbc.Tab(overview, label="Explore dataset", tab_id="tab-explore"),
            dbc.Tab(
                overview_training,
                label="Explore training set",
                tab_id="tab-training"
            ),
            dbc.Tab(aakr_viz, label="AAKR", tab_id="tab-aakr")
        ],
        active_tab="tab-aakr",
    )

    dash_app.layout = html.Div(
        [
            html.Br(),
            card_tabs,
        ],
    )

    # Initialize callbacks after our app is loaded
    # Pass dash_app as a parameter
    init_callbacks(dash_app)

    return dash_app.server


def init_callbacks(app):
    navigate_dataset_callbacks(app)
    training_set_callbacks(app)
    aakr_viz_callbacks(app)
