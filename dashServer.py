import dash
import dash_html_components as html
import dash_core_components as dcc
from dash.dependencies import Input, Output
import dash_bootstrap_components as dbc

import numpy as np
import plotly.graph_objects as go


class dashServer:
    def __init__(self):
        self.app = dash.Dash(__name__, external_stylesheets=[
                             dbc.themes.BOOTSTRAP])
        self.__setLayout()
        self.__configCallbacks()
        self.app.run_server(debug=True)

    def __setLayout(self):
        self.app.layout = html.Div(children=[
            dcc.Graph(id='contour-plot',
                      style={'width': '800px', 'height': '800px'}),

            dbc.Form(
                [
                    dbc.FormGroup(
                        [
                            dbc.Label("Modes per classifier",
                                      className="mr-2"),
                            dbc.Input(type="text", placeholder="Modes"),
                        ],
                        className="mr-3",
                    ),
                    dbc.FormGroup(
                        [
                            dbc.Label("Samples per mode", className="mr-2"),
                            dbc.Input(type="text",
                                      placeholder="Samples"),
                        ],
                        className="mr-3",
                    ),
                    dbc.Button("Generate!", color="primary"),
                ],
                inline=True,
            ),

            dbc.Form(
                [
                    dbc.FormGroup(
                        [
                            dbc.Label("Neural network model",
                                      className="mr-2"),
                            dbc.Input(
                                type="text", placeholder="e.g. [2, 3, 5, 2]"),
                        ],
                        className="mr-3",
                    ),

                    dbc.FormGroup(
                        [
                            dbc.Label("Trainings",
                                      className="mr-2"),
                            dbc.Input(type="text", placeholder="Trainings"),
                        ],
                        className="mr-3",
                    ),

                    dbc.Button("Train and plot!", id="btn-train-and-plot",
                               color="primary", n_clicks=0),
                ],
                inline=True,
            ),

            html.P(id='output-1'),
            html.P(id='output-2'),
        ])

    def __configCallbacks(self):
        print('config callbacks')

        @self.app.callback(
            [Output("output-1", "children")],
            [Input("btn-train-and-plot", "n_clicks")],
        )
        def trainAndPlot(btn_clicks):
            print('btn-train-and-plot', btn_clicks)
            if (btn_clicks % 2 == 0):
                return ['even']
            return ['odd']


dashServer()
