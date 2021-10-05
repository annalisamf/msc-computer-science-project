import dash
import dash_auth  # auth for logging in the app
import dash_core_components as dcc
import dash_html_components as html
import pandas as pd
import plotly.graph_objs as go
from dash.dependencies import Input, Output

################ AUTH FOR LOGIN ###################
USERNAME_PASSWORD_PAIRS = [['afogli01', 'bbk']]

############# DATA ###############
age = pd.read_pickle('/Users/annalisa/PycharmProjects/MSc_Project/dashboard/age.pkl')

gender = pd.read_pickle('/Users/annalisa/PycharmProjects/MSc_Project/dashboard/gender.pkl')

location = pd.read_pickle('/Users/annalisa/PycharmProjects/MSc_Project/dashboard/location.pkl')

topics = pd.read_pickle('/Users/annalisa/PycharmProjects/MSc_Project/dashboard/topics.pkl')

############# STYLE #############
external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']
colors = {
    'background': '#111111',
    'text': '#70263d'
}

graph_colors = {
    'Conservative': '#0087DC',
    'Labour': '#DC241f',
    'LibDem': '#FAA61A'
}

tabs_styles = {
    # 'height': '44px'
    # 'height': '70px'
}
tab_style = {
    'borderBottom': '1px solid #d6d6d6',
    'padding': '6px',
    'fontWeight': 'bold',
    'size': '30px'
}

tab_selected_style = {
    'borderTop': '1px solid #d6d6d6',
    'borderBottom': '1px solid #d6d6d6',
    'backgroundColor': '#70263d',
    'color': 'white',
    'padding': '6px',
    'size': '30px'

}

############# Actual app code ################
app = dash.Dash(__name__, external_stylesheets=external_stylesheets)
auth = dash_auth.BasicAuth(app, USERNAME_PASSWORD_PAIRS)

app.layout = html.Div([
    # children=[
    html.H1(children='An analysis of the 2019 UK General Elections campaign on Twitter',
            style={
                'textAlign': 'center',
                'color': colors['text']
            }),
    html.Div(children='Demographics of retweeters and topics of candidates',
             style={
                 'textAlign': 'center',
                 'color': colors['text']
             }
             ),

    html.Div([html.H6('Choose a party:'),
              dcc.RadioItems(id='party', options=[{'label': party, 'value': party} for party in
                                                  ['Conservative', 'Labour', 'LibDem']],
                             value='Conservative', labelStyle={'display': 'inline-block', 'textAlign': 'center'},

                             )]
             , style={
            # 'display': 'inline-block', 'verticalAlign': 'top', 'width': '30%',
            'textAlign': 'center',
            'padding': '10px'}
             )
    ,

    dcc.Tabs([
        dcc.Tab(label='Demographics of retweeters', style=tab_style, selected_style=tab_selected_style,
                children=[
                    # html.P(),
                    html.Div(
                        [
                            dcc.Graph(id='age',
                                      style={'backgroundColor': colors['background'],
                                             'width': '32%',
                                             'display': 'inline-block',
                                             'border': '1px black solid'}),
                            dcc.Graph(id='gender',
                                      style={'backgroundColor': colors['background'],
                                             'width': '32%',
                                             'display': 'inline-block',
                                             'border': '1px black solid'}),
                            dcc.Graph(id='location',
                                      style={'backgroundColor': colors['background'],
                                             'width': '32%',
                                             'display': 'inline-block',
                                             'border': '1px black solid'})

                        ]
                        , style={'textAlign': 'center', 'padding': '10px'}
                    ),

                ]),
        dcc.Tab(label='Topics tweeted by the candidates', style=tab_style, selected_style=tab_selected_style,
                children=[
                    dcc.Graph(id='topics')
                ]),
    ])
])


@app.callback(Output('age', 'figure'),
              [Input('party', 'value')])
def update_figure(party):
    age_by_party = age[age.party == party]
    return {
        'data': [
            (go.Bar(
                x=age_by_party['predicted'].unique(),
                y=age_by_party['predicted'].value_counts(normalize=True) * 100,
                marker=dict(color=graph_colors[f'{party}'])
            ))
        ],

        'layout': go.Layout(
            title=f'Age of {party}\'s retweeters',
            xaxis={'title': 'Age', 'categoryorder': 'array', 'categoryarray': ['below30', '31-59', 'over60']},
            yaxis={'title': '% of users', 'range': [0, age.predicted.value_counts(normalize=True).max() * 100]},
            hovermode='closest'
        ),

    }


@app.callback(Output('gender', 'figure'),
              [Input('party', 'value')])
def update_figure(party):
    gender_by_party = gender[gender.party == party]
    return {
        'data': [
            go.Pie(
                labels=gender_by_party['predicted'].value_counts().index.tolist(),
                values=gender_by_party['predicted'].value_counts(),
                marker=dict()
            )

        ],

        'layout': go.Layout(
            title=f'Gender of {party}\'s retweeeters',
            hovermode='closest',
            colorway=['lightblue', 'pink']

        )

    }


@app.callback(Output('location', 'figure'),
              [Input('party', 'value')])
def update_figure(party):
    location_by_party = location[location.party == party]
    return {
               'data': [
                   go.Bar(
                       x=location_by_party.county.value_counts()[:10].index.tolist(),
                       y=location_by_party.county.value_counts(normalize=True) * 100,
                       marker=dict(color=graph_colors[f'{party}'])
                   )

               ],

               'layout': go.Layout(
                   title={
                       'text': f'Top 10 counties of {party}\'s retweeters',
                       'font': {
                       }},

            xaxis={
                'title': {
                    'text': 'UK counties',
                    'font': {
                        # 'family': 'Courier New, monospace',
                        'size': 10,
                        # 'color': '#7f7f7f'
                    }
                }
            },
            yaxis={'title': '% of users', 'range': [0, location.county.value_counts(normalize=True).max() * 100]},
            # margin={'l': 30, 'r': 30, 'b': 70, 't': 40}
            margin={'b': 150}

        )
    }


@app.callback(Output('topics', 'figure'),
              [Input('party', 'value')])
def update_figure(party):
    topics_by_party = topics[topics.party == party]
    return {
        'data': [
            (go.Bar(
                x=topics_by_party['topic_name'].unique(),
                y=topics_by_party['docs_perc'],
                marker=dict(color=graph_colors[f'{party}'])
            ))

        ],

        'layout': go.Layout(
            title=f'Topics most tweeted by the {party}\'s candidates',
            xaxis={'title': 'Topics', 'categoryorder': 'total descending'},
            yaxis={'title': '% of candidates', 'range': [0, topics.docs_perc.max()]},
            hovermode='closest',
            margin={'b': 150}
        ),

    }


if __name__ == '__main__':
    app.run_server()
