# pages/Option_Viewer.py

import dash
import dash_bootstrap_components as dbc
from dash import html, dcc, dash_table, callback, Input, Output, State
import requests
import pandas as pd
from io import StringIO
from datetime import datetime
import plotly.graph_objects as go

#######################################################
# 1. Register the page & remove Dash(...) initialization
#######################################################
dash.register_page(__name__, path="/option_viewer", name="Option Data Visualization")


#######################################################
# 2. Constants and Helper Functions
#######################################################
API_URL = "http://127.0.0.1:25510/v2/bulk_hist/option/eod_greeks"

DATA_POINTS = [
    {'label': 'Underlying Price', 'value': 'underlying_price_c'},
    {'label': 'Implied Volatility Call', 'value': 'implied_vol_c'},
    {'label': 'Vega Call', 'value': 'vega_c'},
    {'label': 'Theta Call', 'value': 'theta_c'},
    {'label': 'Delta Call', 'value': 'delta_c'},
    {'label': 'Bid Call', 'value': 'bid_c'},
    {'label': 'Ask Call', 'value': 'ask_c'},
    {'label': 'Implied Volatility Put', 'value': 'implied_vol_p'},
    {'label': 'Vega Put', 'value': 'vega_p'},
    {'label': 'Theta Put', 'value': 'theta_p'},
    {'label': 'Delta Put', 'value': 'delta_p'},
    {'label': 'Bid Put', 'value': 'bid_p'},
    {'label': 'Ask Put', 'value': 'ask_p'},
]


def fetch_eod_greeks_csv(api_url, params):
    """
    Fetch CSV data for EOD Greeks from the given API endpoint.
    """
    try:
        response = requests.get(api_url, params=params, headers={"Accept": "text/csv"})
        response.raise_for_status()
        return response.text
    except requests.exceptions.RequestException as e:
        print(f"API request failed: {e}")
        return None


def process_and_filter_csv(csv_data, desired_strikes):
    """
    Process the CSV data and return a pivoted DataFrame filtered by desired_strikes.
    """
    if not csv_data:
        print("No CSV data received.")
        return None

    csv_io = StringIO(csv_data)
    try:
        df = pd.read_csv(csv_io)

        if 'strike' not in df.columns:
            print("Missing 'strike' column in CSV data.")
            return None
        if 'date' not in df.columns:
            print("Missing 'date' column in CSV data.")
            return None

        # Convert columns appropriately
        df['Date'] = pd.to_datetime(df['date'], format='%Y%m%d')
        df['Strike'] = df['strike'] / 1000

        if 'underlying_price' in df.columns:
            df['underlying_price'] = df['underlying_price'] / 1000
        else:
            print("Missing 'underlying_price' column in CSV data.")

        # Split out underlying_price for calls and puts
        if 'right' in df.columns:
            df['underlying_price_c'] = df.apply(
                lambda row: row['underlying_price'] if row['right'] == 'C' else pd.NA,
                axis=1
            )
            df['underlying_price_p'] = df.apply(
                lambda row: row['underlying_price'] if row['right'] == 'P' else pd.NA,
                axis=1
            )
        else:
            print("Missing 'right' column in CSV data.")
            return None

        # Filter the desired strikes
        df_filtered = df[df['Strike'].isin(desired_strikes)]
        if df_filtered.empty:
            print(f"No data found for strike prices {desired_strikes}.")
            return None

        df_filtered['Strike'] = df_filtered['Strike'].astype(str)

        # Pivot calls & puts
        df_pivot = df_filtered.pivot_table(
            index=['Date', 'Strike'],
            columns='right',
            aggfunc='first'
        )

        # Flatten the pivoted columns
        df_pivot.columns = [f"{col[0].lower()}_{col[1].lower()}" for col in df_pivot.columns]
        df_pivot.reset_index(inplace=True)

        # Reorder / fill missing columns
        desired_columns = [
            'Date', 'Strike', 'underlying_price_c', 'implied_vol_c',
            'vega_c', 'theta_c', 'delta_c', 'bid_c', 'ask_c',
            'bid_p', 'ask_p', 'delta_p', 'theta_p', 'vega_p', 'implied_vol_p'
        ]

        # Create missing columns if they don't exist
        for col in desired_columns:
            if col not in df_pivot.columns:
                df_pivot[col] = pd.NA

        df_pivot = df_pivot[desired_columns]
        return df_pivot

    except Exception as e:
        print(f"Error processing CSV data: {e}")
        return None


#######################################################
# 3. Define the module-level layout
#######################################################
layout = dbc.Container(
    [
        dbc.NavbarSimple(
            brand="Option Data Visualization Dashboard",
            color="primary",
            dark=True,
            sticky="top",
        ),
        # First Row: Input Parameters (left) and Graph (right)
        dbc.Row(
            [
                dbc.Col(
                    dbc.Card(
                        [
                            dbc.CardHeader("Input Parameters"),
                            dbc.CardBody(
                                [
                                    dbc.Form(
                                        [
                                            dbc.Label("Root Symbol"),
                                            dbc.Input(
                                                id='optdata-root-symbol',
                                                type='text',
                                                value='AAPL',
                                                placeholder='Enter root symbol'
                                            ),
                                        ],
                                        className="mb-3"
                                    ),
                                    dbc.Row(
                                        [
                                            dbc.Col(
                                                dbc.Form(
                                                    [
                                                        dbc.Label("Expiration Date"),
                                                        dcc.DatePickerSingle(
                                                            id='optdata-exp-date',
                                                            date=datetime(2025, 2, 14),
                                                            display_format='YYYY-MM-DD',
                                                            style={'width': '100%'}
                                                        ),
                                                    ]
                                                ),
                                                width=4
                                            ),
                                            dbc.Col(
                                                dbc.Form(
                                                    [
                                                        dbc.Label("Start Date"),
                                                        dcc.DatePickerSingle(
                                                            id='optdata-start-date',
                                                            date=datetime(2025, 2, 1),
                                                            display_format='YYYY-MM-DD',
                                                            style={'width': '100%'}
                                                        ),
                                                    ]
                                                ),
                                                width=4
                                            ),
                                            dbc.Col(
                                                dbc.Form(
                                                    [
                                                        dbc.Label("End Date"),
                                                        dcc.DatePickerSingle(
                                                            id='optdata-end-date',
                                                            date=datetime(2025, 2, 14),
                                                            display_format='YYYY-MM-DD',
                                                            style={'width': '100%'}
                                                        ),
                                                    ]
                                                ),
                                                width=4
                                            ),
                                        ],
                                        className="mb-3"
                                    ),
                                    dbc.Form(
                                        [
                                            dbc.Label("Desired Strike Prices ($)"),
                                            dbc.Input(
                                                id='optdata-desired-strikes',
                                                type='text',
                                                value='230',
                                                placeholder='Enter strike prices separated by commas'
                                            ),
                                            dbc.FormText("Example: 220, 230, 240"),
                                        ],
                                        className="mb-3"
                                    ),
                                    dbc.Form(
                                        [
                                            dbc.Label("Data Points to Plot"),
                                            dcc.Dropdown(
                                                id='optdata-data-points',
                                                options=DATA_POINTS,
                                                value=['delta_c'],
                                                multi=True,
                                                clearable=False
                                            ),
                                        ],
                                        className="mb-3"
                                    ),
                                    dbc.Button(
                                        'Submit',
                                        id='optdata-submit-button',
                                        color='primary',
                                        className='w-100',
                                        n_clicks=0
                                    ),
                                    dcc.Store(id='optdata-stored-data'),
                                ]
                            )
                        ],
                        className='h-100'
                    ),
                    md=4
                ),
                dbc.Col(
                    dbc.Card(
                        [
                            dbc.CardHeader("Option Data Graph"),
                            dbc.CardBody(
                                [
                                    dcc.Loading(
                                        id="optdata-loading-graph",
                                        type="default",
                                        children=dcc.Graph(id='optdata-data-graph')
                                    )
                                ],
                                style={'height': '100%'}
                            )
                        ],
                        className='h-100'
                    ),
                    md=8
                ),
            ],
            className='align-items-stretch',
            style={'marginTop': '20px'}
        ),
        # Second Row: Data Table
        dbc.Row(
            [
                dbc.Col(
                    dbc.Card(
                        [
                            dbc.CardHeader("Data Table"),
                            dbc.CardBody(
                                [
                                    dcc.Loading(
                                        id="optdata-loading-table",
                                        type="default",
                                        children=dash_table.DataTable(
                                            id='optdata-data-table',
                                            columns=[],
                                            data=[],
                                            style_table={'overflowX': 'auto'},
                                            style_cell={
                                                'minWidth': '80px',
                                                'width': '80px',
                                                'maxWidth': '180px',
                                                'whiteSpace': 'normal',
                                                'textAlign': 'center'
                                            },
                                            style_data_conditional=[
                                                {
                                                    'if': {'column_type': 'numeric'},
                                                    'textAlign': 'right',
                                                },
                                            ],
                                        )
                                    )
                                ]
                            )
                        ]
                    ),
                    md=12
                ),
            ],
            style={'marginTop': '20px'}
        ),
    ],
    fluid=True
)


#######################################################
# 4. Callbacks using @dash.callback
#######################################################

# A) Fetch and store the DataFrame JSON when the user clicks "Submit"
@callback(
    Output('optdata-stored-data', 'data'),
    Input('optdata-submit-button', 'n_clicks'),
    State('optdata-root-symbol', 'value'),
    State('optdata-exp-date', 'date'),
    State('optdata-start-date', 'date'),
    State('optdata-end-date', 'date'),
    State('optdata-desired-strikes', 'value')
)
def fetch_and_store_data(n_clicks, root_symbol, exp_date, start_date_, end_date_, desired_strikes):
    if n_clicks < 1:
        return None

    # Convert date strings to YYYYMMDD
    try:
        exp_dt = datetime.strptime(exp_date, '%Y-%m-%d')
        start_dt = datetime.strptime(start_date_, '%Y-%m-%d')
        end_dt = datetime.strptime(end_date_, '%Y-%m-%d')
    except (TypeError, ValueError):
        return None

    exp_date_int = int(exp_dt.strftime('%Y%m%d'))
    start_date_int = int(start_dt.strftime('%Y%m%d'))
    end_date_int = int(end_dt.strftime('%Y%m%d'))

    if not desired_strikes.strip():
        return None
    desired_strikes_list = [float(s.strip()) for s in desired_strikes.split(',')]

    params = {
        "root": root_symbol.upper(),
        "exp": exp_date_int,
        "start_date": start_date_int,
        "end_date": end_date_int,
        "use_csv": True
    }

    csv_response = fetch_eod_greeks_csv(API_URL, params)
    if not csv_response:
        return None

    df_filtered = process_and_filter_csv(csv_response, desired_strikes_list)
    if df_filtered is not None:
        return df_filtered.to_json(date_format='iso', orient='split')
    else:
        return None


# B) Update the graph and table whenever the DataFrame JSON or chosen data points change
@callback(
    Output('optdata-data-graph', 'figure'),
    Output('optdata-data-table', 'columns'),
    Output('optdata-data-table', 'data'),
    Input('optdata-stored-data', 'data'),
    Input('optdata-data-points', 'value'),
    State('optdata-root-symbol', 'value')
)
def update_graph_and_table(df_json, data_points, root_symbol):
    if df_json is None or not data_points:
        # Return an empty figure and empty table
        empty_fig = go.Figure()
        empty_fig.update_layout(
            title="Option Data Visualization",
            xaxis_title="Date",
            yaxis_title="Value"
        )
        return empty_fig, [], []

    # Reconstruct DataFrame
    df = pd.read_json(df_json, orient='split')

    fig = go.Figure()

    # Decide whether to plot 'underlying_price_c' on a secondary axis
    if 'underlying_price_c' in data_points:
        data_points_without_underlying = [dp for dp in data_points if dp != 'underlying_price_c']
        plot_secondary_y = True
    else:
        data_points_without_underlying = data_points
        plot_secondary_y = False

    # Plot each chosen data point vs. time, grouped by Strike
    for data_point in data_points_without_underlying:
        if data_point not in df.columns:
            continue
        for strike in df['Strike'].unique():
            df_sub = df[df['Strike'] == strike][['Date', data_point]].dropna()
            fig.add_trace(go.Scatter(
                x=df_sub['Date'],
                y=df_sub[data_point],
                mode='lines+markers',
                name=f"{data_point} (Strike {strike})"
            ))

    if plot_secondary_y and 'underlying_price_c' in df.columns:
        # Plot underlying price on the secondary y-axis
        under_df = df[['Date', 'underlying_price_c']].dropna().drop_duplicates(subset='Date')
        fig.add_trace(go.Scatter(
            x=under_df['Date'],
            y=under_df['underlying_price_c'],
            mode='lines+markers',
            name='Underlying Price',
            yaxis='y2',
            line=dict(color='black', width=2, dash='dash')
        ))

        fig.update_layout(
            yaxis=dict(title='Data Points'),
            yaxis2=dict(
                title='Underlying Price',
                overlaying='y',
                side='right',
                showgrid=False
            ),
            xaxis=dict(title='Date'),
            legend=dict(orientation="h"),
            title=f"Option Data Over Time for {root_symbol.upper()}"
        )
    else:
        fig.update_layout(
            yaxis=dict(title='Data Points'),
            xaxis=dict(title='Date'),
            legend=dict(orientation="h"),
            title=f"Option Data Over Time for {root_symbol.upper()}"
        )

    # Prepare the DataTable
    df_display = df.copy()
    df_display['Date'] = df_display['Date'].dt.strftime('%Y-%m-%d')
    numeric_cols = df_display.select_dtypes('number').columns
    df_display[numeric_cols] = df_display[numeric_cols].round(2)

    columns = [{"name": col, "id": col} for col in df_display.columns]
    data = df_display.to_dict('records')

    return fig, columns, data
