from dash import Dash, html, dcc
from dash.dependencies import Input, Output
import requests
import plotly.express as px
import pandas as pd

# Initialize Dash app
app = Dash(__name__, server=False)

# Fetch data from Flask backend
def fetch_summary_stats():
    return requests.get("http://127.0.0.1:5000/api/summary_stats").json()

def fetch_fraud_trends():
    return requests.get("http://127.0.0.1:5000/api/fraud_trends").json()

def fetch_fraud_by_geography():
    return requests.get("http://127.0.0.1:5000/api/fraud_by_geography").json()

def fetch_fraud_by_device_browser():
    return requests.get("http://127.0.0.1:5000/api/fraud_by_device_browser").json()

# Dashboard layout
app.layout = html.Div([
    html.H1("Fraud Detection Dashboard"),

    # Summary boxes
    html.Div(id='summary-boxes'),

    # Line chart for fraud trends
    html.Div([
        html.H3("Fraud Cases Over Time"),
        dcc.Graph(id='line-chart-fraud-trends')
    ]),

    # Geography chart
    html.Div([
        html.H3("Fraud by Geography"),
        dcc.Graph(id='geo-chart')
    ]),

    # Device and Browser chart
    html.Div([
        html.H3("Fraud by Device and Browser"),
        dcc.Graph(id='device-browser-chart')
    ])
])

# Callbacks to update the dashboard
@app.callback(
    [Output('summary-boxes', 'children'),
     Output('line-chart-fraud-trends', 'figure'),
     Output('geo-chart', 'figure'),
     Output('device-browser-chart', 'figure')],
    [Input('interval-component', 'n_intervals')]
)
def update_dashboard(n):
    # Fetch data from the backend
    summary_stats = fetch_summary_stats()
    fraud_trends = fetch_fraud_trends()
    fraud_by_geography = fetch_fraud_by_geography()
    fraud_by_device_browser = fetch_fraud_by_device_browser()

    # Summary boxes layout
    summary_layout = html.Div([
        html.Div([
            html.H4("Total Transactions"),
            html.P(summary_stats['total_transactions'])
        ]),
        html.Div([
            html.H4("Fraud Cases"),
            html.P(summary_stats['fraud_cases'])
        ]),
        html.Div([
            html.H4("Fraud Percentage"),
            html.P(f"{summary_stats['fraud_percentage']:.2f}%")
        ])
    ], style={'display': 'flex', 'justify-content': 'space-around'})

    # Line chart for fraud trends
    fraud_trend_df = pd.DataFrame(fraud_trends)
    line_chart = px.line(fraud_trend_df, x="date", y="fraud_cases", title="Fraud Cases Over Time")

    # Geography chart
    geo_df = pd.DataFrame(fraud_by_geography)
    geo_chart = px.choropleth(geo_df, locations="country", locationmode='country names', color="class",
                              title="Fraud by Geography", color_continuous_scale="reds")

    # Device and Browser chart
    device_browser_df = pd.DataFrame(fraud_by_device_browser)
    device_browser_chart = px.bar(device_browser_df, x="device", y="class", color="browser",
                                  title="Fraud by Device and Browser")

    return summary_layout, line_chart, geo_chart, device_browser_chart

if __name__ == "__main__":
    app.run_server(debug=True, port=8050)