import dash
import dash_html_components as html
import dash_core_components as dcc
from dash.dependencies import Input, Output, State
import dash_bootstrap_components as dbc

import numpy as np
import plotly.graph_objects as go
import classifier


class dashServer:
    def __init__(self):
        self.classifier1 = None
        self.classifier2 = None
        self.network = None

        self.app = dash.Dash(__name__, external_stylesheets=[
                             dbc.themes.BOOTSTRAP])
        self.__setLayout()
        self.__configCallbacks()
        self.app.run_server(debug=True)

    def __setLayout(self):
        self.app.layout = html.Div(children=[
            dcc.Graph(id='contour-plot', figure={},
                      style={'width': '800px', 'height': '800px'}),

            dbc.Form(
                [
                    dbc.FormGroup(
                        [
                            dbc.Label('Modes per classifier',
                                      className='mr-2'),
                            dbc.Input(
                                type='number', id='input-modes-per-classifier', placeholder='Modes'),
                        ],
                        className='mr-3',
                    ),
                    dbc.FormGroup(
                        [
                            dbc.Label('Samples per mode', className='mr-2'),
                            dbc.Input(
                                type='number', id='input-samples-per-mode', placeholder='Samples'),
                        ],
                        className='mr-3',
                    ),
                    dbc.Button('Generate!', id='btn-generate-classifiers',
                               color='primary', n_clicks_timestamp='0'),
                ],
                inline=True,
            ),

            dbc.Form(
                [
                    dbc.FormGroup(
                        [
                            dbc.Label('Neural network model',
                                      className='mr-2'),
                            dbc.Input(
                                type='text', id='input-network-model', placeholder='e.g. [2, 3, 5, 2]'),
                        ],
                        className='mr-3',
                    ),

                    dbc.FormGroup(
                        [
                            dbc.Label('Epochs',
                                      className='mr-2'),
                            dbc.Input(type='number', id='input-epochs',
                                      placeholder='Epochs'),
                        ],
                        className='mr-3',
                    ),

                    dbc.Button('Train and plot!', id='btn-train-and-plot',
                               color='primary', n_clicks_timestamp='0'),
                ],
                inline=True,
            ),

            html.P(id='output-1'),
            html.P(id='output-2'),
        ])

    def __configCallbacks(self):
        print('config callbacks')

        # @self.app.callback(
        #     [Output('contour-plot', 'figure'),
        #      Output('btn-train-and-plot', 'disabled')],
        #     [Input('btn-generate-classifiers', 'n_clicks')],
        #     [State('input-modes-per-classifier', 'value'),
        #      State('input-samples-per-mode', 'value')],
        # )
        # def gernerateClassifiers(btn_clicks, modesPerClassifier, samplesPerMode):
        #     if (btn_clicks == 0 or modesPerClassifier in [None, 0] or samplesPerMode in [None, 0]):
        #         return [{}, True]

        #     c1 = classifier.Classifier(modesPerClassifier, samplesPerMode)
        #     c2 = classifier.Classifier(modesPerClassifier, samplesPerMode)

        #     self.classifier1 = c1
        #     self.classifier2 = c2

        #     scatter1 = go.Scatter(
        #         x=c1.getAllSamples()[0],
        #         y=c1.getAllSamples()[1],
        #         name=0,
        #         mode='markers'
        #     )

        #     scatter2 = go.Scatter(
        #         x=c2.getAllSamples()[0],
        #         y=c2.getAllSamples()[1],
        #         name=1,
        #         mode='markers'
        #     )

        #     fig = go.Figure(data=[scatter1, scatter2])
        #     fig.update_xaxes(range=[-.1, 1.1])
        #     fig.update_yaxes(range=[-.1, 1.1])

        #     return [fig, False]

        # @self.app.callback(
        #     Output('contour-plot', 'figure'),
        #     [Input('btn-train-and-plot', 'n_clicks')],
        #     [State('input-network-model', 'value'),
        #      State('input-epochs', 'value')],
        # )
        # def trainAndPlot(btn_clicks, networkModel, epochs):
        #     print('btn-train-and-plot', btn_clicks, networkModel, epochs)
        #     return {}

        @self.app.callback(
            Output('contour-plot', 'figure'),
            [Input('btn-generate-classifiers', 'n_clicks_timestamp'),
             Input('btn-train-and-plot', 'n_clicks_timestamp')],
            [State('input-modes-per-classifier', 'value'),
             State('input-samples-per-mode', 'value')]
        )
        def whichButtonWasClicked(btn_generate_timestamp, btn_training_timestamp, modesPerClassifier, samplesPerMode):
            operation = ''

            if (isinstance(btn_generate_timestamp, str) and isinstance(btn_training_timestamp, int)):
                operation = 'training'
            elif (isinstance(btn_training_timestamp, str) and isinstance(btn_generate_timestamp, int)):
                operation = 'generate'
            elif (btn_training_timestamp < btn_generate_timestamp):
                operation = 'generate'
            elif (btn_generate_timestamp < btn_training_timestamp):
                operation = 'training'

            if (operation == ''):
                return {}
            elif (operation == 'generate'):
                if (modesPerClassifier is None or modesPerClassifier <= 0 or samplesPerMode is None or samplesPerMode <= 0):
                    return {}

                c1 = classifier.Classifier(modesPerClassifier, samplesPerMode)
                c2 = classifier.Classifier(modesPerClassifier, samplesPerMode)

                self.classifier1 = c1
                self.classifier2 = c2

                scatter1 = go.Scatter(
                    x=c1.getAllSamples()[0],
                    y=c1.getAllSamples()[1],
                    name=0,
                    mode='markers'
                )

                scatter2 = go.Scatter(
                    x=c2.getAllSamples()[0],
                    y=c2.getAllSamples()[1],
                    name=1,
                    mode='markers'
                )

                fig = go.Figure(data=[scatter1, scatter2])
                fig.update_xaxes(range=[-.1, 1.1])
                fig.update_yaxes(range=[-.1, 1.1])

                return fig

            print(operation)
            return {}


dashServer()
