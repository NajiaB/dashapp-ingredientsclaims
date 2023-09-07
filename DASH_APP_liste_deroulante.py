# -*- coding: utf-8 -*-
"""
Created on Wed Aug 16 20:06:32 2023

@author: BOUADDOUCH Najia
"""

#!pip install dash
#!pip install dash-table
import dash
from dash  import dcc
from dash import html
from dash.dependencies import Input, Output, State
import pickle
import os
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from dash.exceptions import PreventUpdate
import dash_table
import webbrowser
from threading import Timer
app = dash.Dash(__name__)

ingredients = vectorizer.get_feature_names_out()

ingredient_options = [{'label': ingredient, 'value': ingredient} for ingredient in ingredients]

app.layout = html.Div([
    html.H1("Claims Predictor"),
    dcc.Tabs([
        dcc.Tab(label='Predict Claims', children=[
            dcc.Dropdown(
                id="ingredientDropdown",
                options=ingredient_options,
                multi=True,
                placeholder="Select Ingredients",
            ),
            html.Div(id="selectedIngredients", children=[]),
            html.Button("Predict", id="predictButton"),
            dcc.Graph(id="predictionPlot", figure={}),
        ]),
        dcc.Tab(label='Importance of Ingredients in Prediction', children=[
            dcc.Dropdown(
                id="claimDropdown",
                options=[{'label': label, 'value': label} for label in labels],  
                placeholder="Select a Claim",
            ),
            html.Div(id="variableImportanceTable"),
        ]),
    ]),
])


@app.callback(
    [Output("selectedIngredients", "children"), Output("predictionPlot", "figure")],
    [Input("predictButton", "n_clicks")],
    [State("ingredientDropdown", "value")])



def update_plot(n_clicks, selected_ingredients):
    print("Callback triggered")
    if n_clicks is None :
        raise PreventUpdate
   
    print("Selected Ingredients:", selected_ingredients)
    if selected_ingredients:
        
        classifiers = load_all_classifiers(labels)
        vectorizer = load_vectorizer(chemin + 'binary_models')
        selected_ingredients_str = ', '.join(selected_ingredients) #etape primordiale car notre get_prediction() prend une chaine de caracteres, et non une liste
        predicted_probabilities = get_prediction(selected_ingredients_str, vectorizer, classifiers, labels)
    
        #
        data = {'Labels': labels, 'Probabilities': predicted_probabilities[0].tolist()[0]}
        df = pd.DataFrame(data)
        result = df.groupby(["Labels"])['Probabilities'].median().reset_index().sort_values('Probabilities')
        
        figure = {
            'data': [
                {'x': result['Labels'], 'y': result['Probabilities'], 'type': 'bar', 'marker': {
                    'color': ['green' if p > 0.6 else 'blue' for p in result['Probabilities']]
                }}
            ],
            'layout': {
                'xaxis': {'title': 'Labels'},
                'yaxis': {'title': 'Probabilities'},
                'title': 'Predicted Probabilities of Claims',
                'xaxis_tickangle': -45,
                'shapes': [
                    {
                        'type': 'line',
                        'x0': result['Labels'].iloc[0],
                        'x1': result['Labels'].iloc[-1],
                        'y0': 0.5,
                        'y1': 0.5,
                        'line': {
                            'color': 'red',
                            'width': 2,
                            'dash': 'dash'
                        }
                    }
                ]
            }
        }
    else:
        # Provide an empty figure if no ingredients are selected
        figure = {}
        

    return [], figure 

@app.callback(
    Output("variableImportanceTable", "children"),
    Input("claimDropdown", "value"),
    State("ingredientDropdown", "value"))  # Adding State for selected ingredients
def update_importance_table(selected_claim, selected_ingredients):
    if selected_claim is None:
        return html.Div("Please select a claim.")
    
    if not selected_ingredients:
        return html.Div("Please select ingredients.")
    
    test_term_doc = vectorizer.transform(selected_ingredients)
    
    df_words_weights = get_local_weights_df(vectorizer, test_term_doc, classifiers, selected_claim)
    df_words_weights['weights'] = round(df_words_weights['weights'],2)
    table = dash_table.DataTable(
        columns=[{"name": col, "id": col} for col in df_words_weights.columns],
        data=df_words_weights.to_dict("records"),
        style_table={"overflowX": "auto"},  # Enable horizontal scrolling
        sort_action="native",  # Enable sorting
        sort_mode="multi",  # Enable multi-column sorting
    )
    
    return table

if __name__ == "__main__":
    app.run_server(debug=False)


port = 8050 # or simply open on the default `8050` port

def open_browser():
	webbrowser.open_new("http://localhost:{}".format(port))

if __name__ == '__main__':
    Timer(1, open_browser).start();
    app.run_server(debug=True, port=port)