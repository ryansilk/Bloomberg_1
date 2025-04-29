import dash
import dash_bootstrap_components as dbc
import requests
from datetime import datetime, date
import pandas as pd
from dash import dcc, html, dash_table, callback, Input, Output, State
from io import StringIO
import plotly.graph_objects as go

#########################################
# 1. Register Page
#########################################
dash.register_page(__name__, path="/option_mon", title="Option Monitor")


#########################################
# 2. Helper Functions for Option Chain
#########################################
def format_date(yyyymmdd):
    """Converts a YYYYMMDD string to a datetime object, or None on error."""
    try:
        return datetime.strptime(str(yyyymmdd), '%Y%m%d')
    except ValueError:
        return None


def list_expirations(root):
    """
    Fetches the list of expiration dates for a given root (stock) from the API.
    Returns a list of expiration strings (YYYYMMDD).
    """
    base_url = "http://127.0.0.1:25510/v2/list/expirations"
    headers = {'Accept': 'application/json'}
    params = {'root': root}
    response = requests.get(base_url, headers=headers, params=params)
    response.raise_for_status()
    data = response.json()
    if data.get('header', {}).get('error_type') not in (None, "null"):
        msg = data['header'].get('error_msg', 'Unknown error')
        raise Exception(f"Error fetching expirations: {msg}")
    expirations = data.get('response', [])
    if not expirations:
        raise Exception(f"No expirations found for root '{root}'.")
    return expirations


def fetch_option_chain(root, expiration):
    """
    Fetches the option chain for a given stock and expiration date from the API.
    Returns a DataFrame.
    """
    base_url = "http://127.0.0.1:25510/v2/bulk_snapshot/option/greeks"
    headers = {'Accept': 'text/csv'}
    params = {
        'root': root,
        'exp': expiration,
        'use_csv': 'true'
    }
    response = requests.get(base_url, headers=headers, params=params)
    response.raise_for_status()
    df = pd.read_csv(StringIO(response.text))
    if df.empty:
        raise Exception(f"No option data found for '{root}' with expiration '{expiration}'.")
    return df


def detect_scaling_factor(df):
    """
    Detects the scaling factor for the 'strike' column.
    Checks several possible factors to see which yields a 'reasonable' strike range.
    """
    possible_factors = [1, 10, 100, 1000, 10000]
    for factor in possible_factors:
        scaled = df['strike'] / factor
        if scaled.between(1, 10000).all():
            return factor
    raise Exception("Unable to detect scaling factor for 'strike' prices.")


def build_option_monitor_table(df_raw, center_strike, num_strikes, option_type="both"):
    """
    Processes the raw option chain data to build a merged table that mimics
    the Bloomberg Terminal's option monitor layout.
    It assumes the raw CSV has columns such as 'strike', 'right', 'bid', 'ask', 'last',
    'implied_vol', and 'volume'. The resulting table will have the call data on the left,
    the strike (center) in the middle and the put data on the right.
    The table is filtered to show a total number of rows (num_strikes) centered around the
    specified center strike.
    The option_type parameter determines if only calls or only puts (or both) are shown.
    """
    try:
        scale = detect_scaling_factor(df_raw)
    except Exception:
        scale = 1

    df_raw['strike'] = df_raw['strike'].astype(float) / scale

    # Separate calls and puts
    calls_df = df_raw[df_raw['right'] == 'C'].copy()
    puts_df = df_raw[df_raw['right'] == 'P'].copy()

    # Rename the columns to common names for each side
    calls_df.rename(columns={
        'bid': 'Call Bid',
        'ask': 'Call Ask',
        'last': 'Call Last',
        'implied_vol': 'IV Call',
        'volume': 'VOLM Call'
    }, inplace=True)
    puts_df.rename(columns={
        'bid': 'Put Bid',
        'ask': 'Put Ask',
        'last': 'Put Last',
        'implied_vol': 'IV Put',
        'volume': 'VOLM Put'
    }, inplace=True)

    calls_cols = ['strike', 'Call Bid', 'Call Ask', 'Call Last', 'IV Call', 'VOLM Call']
    puts_cols = ['strike', 'Put Bid', 'Put Ask', 'Put Last', 'IV Put', 'VOLM Put']

    calls_df = calls_df[calls_cols]
    puts_df = puts_df[puts_cols]

    # Merge calls and puts on strike (outer join so missing data on either side is preserved)
    merged_df = pd.merge(calls_df, puts_df, on='strike', how='outer')
    merged_df.rename(columns={'strike': 'Strike'}, inplace=True)
    merged_df.sort_values(by='Strike', inplace=True)
    merged_df.reset_index(drop=True, inplace=True)

    # Filter rows to show only a window of strikes centered around the provided center strike
    if not merged_df.empty:
        differences = (merged_df['Strike'] - center_strike).abs()
        closest_index = differences.idxmin()
        start_index = max(0, closest_index - num_strikes // 2)
        end_index = start_index + num_strikes
        filtered_df = merged_df.iloc[start_index:end_index]
    else:
        filtered_df = merged_df

    # Drop the side not selected if needed
    if option_type == "calls":
        filtered_df = filtered_df.drop(columns=['Put Bid', 'Put Ask', 'Put Last', 'IV Put', 'VOLM Put'],
                                       errors='ignore')
    elif option_type == "puts":
        filtered_df = filtered_df.drop(columns=['Call Bid', 'Call Ask', 'Call Last', 'IV Call', 'VOLM Call'],
                                       errors='ignore')

    filtered_df.reset_index(drop=True, inplace=True)
    return filtered_df


#########################################
# 2B. Helper Functions for Stock Price
#########################################
def fetch_stock_eod(ticker, start_date, end_date):
    """
    Fetches historical end-of-day stock data for the given ticker between start_date and end_date.
    Expects CSV data.
    """
    url = "http://127.0.0.1:25510/v2/hist/stock/eod"
    params = {
        "root": ticker.upper(),
        "start_date": start_date,
        "end_date": end_date,
        "use_csv": "true"
    }
    response = requests.get(url, params=params)
    response.raise_for_status()
    df = pd.read_csv(StringIO(response.text))
    return df


def get_stock_price_changes(ticker):
    """
    Fetches historical stock data and returns the current price along with percentage changes
    over the previous 5 days, 1 month, 6 months, and 1 year.
    """
    end_date = datetime.today().strftime("%Y%m%d")
    start_date = (datetime.today() - pd.Timedelta(days=500)).strftime("%Y%m%d")
    df = fetch_stock_eod(ticker, start_date, end_date)
    df['date'] = pd.to_datetime(df['date'], format='%Y%m%d')
    df.sort_values(by='date', inplace=True)
    current_price = df.iloc[-1]['close']
    changes = {}
    timeframes = {
        "5 Days": 5,
        "1 Month": 22,
        "6 Months": 126,
        "1 Year": 252
    }
    for label, days in timeframes.items():
        if len(df) > days:
            past_price = df.iloc[-(days + 1)]['close']
            pct_change = ((current_price - past_price) / past_price) * 100
            changes[label] = round(pct_change, 2)
        else:
            changes[label] = "N/A"
    return current_price, changes


#########################################
# 3. Page Layout
#########################################
layout = dbc.Container([
    dbc.Row([
        dbc.Col([
            dbc.Label("Enter Ticker Symbol:"),
            dbc.Input(
                id="option-ticker-input",
                type="text",
                placeholder="e.g., TSLA",
                value=""
            )
        ], md=2),
        dbc.Col([
            dbc.Label("Center Strike"),
            dbc.Input(
                id="option-center-strike-input",
                type="number",
                placeholder="Enter Center Strike",
                value=20
            )
        ], md=2),
        dbc.Col([
            dbc.Label("Number of Strikes"),
            dbc.Input(
                id="option-num-strikes-input",
                type="number",
                placeholder="Enter # of Strikes",
                value=20
            )
        ], md=2),
        dbc.Col([
            dbc.Label("Expiration"),
            dcc.Dropdown(
                id="option-expiration-dropdown",
                options=[],  # populated via callback
                placeholder="Select Expiration"
            )
        ], md=3),
        dbc.Col([
            dbc.Label("As Of Date"),
            dcc.DatePickerSingle(
                id="option-asof-date-picker",
                date=date.today()
            )
        ], md=3)
    ], className="mb-3"),
    dbc.Row([
        dbc.Col(
            dbc.RadioItems(
                id="option-type-radio",
                options=[
                    {"label": "Both", "value": "both"},
                    {"label": "Calls Only", "value": "calls"},
                    {"label": "Puts Only", "value": "puts"}
                ],
                value="both",
                inline=True
            ),
            width=12
        )
    ], className="mb-3"),
    # Option Monitor Header Row
    dbc.Row([
        dbc.Col(html.Div("Center Strike", style={"fontWeight": "bold", "textAlign": "center"}), md=2),
        dbc.Col(html.Div("Calls/Puts", style={"fontWeight": "bold", "textAlign": "center"}), md=2),
        dbc.Col(html.Div("Calls", style={"fontWeight": "bold", "textAlign": "center"}), md=2),
        dbc.Col(html.Div("Puts", style={"fontWeight": "bold", "textAlign": "center"}), md=2),
        dbc.Col(html.Div("Term Structure", style={"fontWeight": "bold", "textAlign": "center"}), md=2),
        dbc.Col(html.Div("Moneyness", style={"fontWeight": "bold", "textAlign": "center"}), md=2)
    ], className="mb-3"),
    # Main Option Monitor Table
    dbc.Row(
        dbc.Col(
            html.Div(id="option-monitor-table-div")
        )
    ),
    # Stock Price Info
    dbc.Row(
        dbc.Col(
            html.Div(id="option-stock-info-div")
        ),
        className="mt-4"
    )
], fluid=True, className="pt-4")


#########################################
# 4. Callbacks
#########################################

# Callback to update the stock info (current price and percentage changes)
@callback(
    Output("option-stock-info-div", "children"),
    Input("option-ticker-input", "value")
)
def update_stock_info(ticker_symbol):
    if not ticker_symbol:
        return ""
    try:
        current_price, changes = get_stock_price_changes(ticker_symbol)
    except Exception as e:
        return html.Div(f"Error fetching stock info: {e}", style={"color": "red"})
    table = html.Table(
        [
            html.Thead(
                html.Tr([
                    html.Th("Current Price"),
                    html.Th("5 Days Change"),
                    html.Th("1 Month Change"),
                    html.Th("6 Months Change"),
                    html.Th("1 Year Change")
                ]),
                style={
                    "backgroundColor": "#f8f9fa",
                    "border": "1px solid #ddd",
                    "textAlign": "center",
                    "padding": "8px"
                }
            ),
            html.Tbody(
                html.Tr([
                    html.Td(f"${current_price:.2f}", style={"padding": "8px", "border": "1px solid #ddd"}),
                    html.Td(f"{changes.get('5 Days', 'N/A')}%", style={"padding": "8px", "border": "1px solid #ddd"}),
                    html.Td(f"{changes.get('1 Month', 'N/A')}%", style={"padding": "8px", "border": "1px solid #ddd"}),
                    html.Td(f"{changes.get('6 Months', 'N/A')}%", style={"padding": "8px", "border": "1px solid #ddd"}),
                    html.Td(f"{changes.get('1 Year', 'N/A')}%", style={"padding": "8px", "border": "1px solid #ddd"}),
                ]),
                style={"textAlign": "center"}
            )
        ],
        style={"width": "100%", "borderCollapse": "collapse", "marginBottom": "20px"}
    )
    return table


# Callback to populate the Expiration Dropdown based on the ticker symbol.
@callback(
    Output("option-expiration-dropdown", "options"),
    Input("option-ticker-input", "value")
)
def update_expiration_options(ticker):
    if not ticker:
        return []
    try:
        exps = list_expirations(ticker.upper())
        options = [{"label": format_date(exp).strftime("%Y-%m-%d") if format_date(exp) else exp, "value": exp} for exp
                   in exps if format_date(exp)]
        return options
    except Exception as e:
        return [{"label": str(e), "value": ""}]


# Callback to build the option monitor table based on user inputs.
@callback(
    Output("option-monitor-table-div", "children"),
    Input("option-ticker-input", "value"),
    Input("option-center-strike-input", "value"),
    Input("option-num-strikes-input", "value"),
    Input("option-expiration-dropdown", "value"),
    Input("option-asof-date-picker", "date"),
    Input("option-type-radio", "value")
)
def update_option_monitor_table(ticker, center_strike, num_strikes, expiration, asof_date, option_type):
    if not ticker or not expiration or center_strike is None or num_strikes is None:
        return ""
    try:
        # The API does not currently support the "as of" date parameter.
        df_raw = fetch_option_chain(ticker.upper(), expiration)
        merged_df = build_option_monitor_table(df_raw, center_strike, num_strikes, option_type)
    except Exception as e:
        return html.Div(f"Error fetching options: {e}", style={"color": "red"})

    columns = [{"name": col, "id": col} for col in merged_df.columns]
    data = merged_df.to_dict("records")
    table = dash_table.DataTable(
        columns=columns,
        data=data,
        fixed_rows={'headers': True},
        style_table={
            "maxHeight": "500px",
            "overflowY": "auto",
            "overflowX": "auto",
            "border": "1px solid #ddd"
        },
        style_cell={
            "textAlign": "center",
            "padding": "8px",
            "minWidth": "80px",
            "whiteSpace": "normal"
        },
        style_header={
            "backgroundColor": "#f8f9fa",
            "fontWeight": "bold",
            "border": "1px solid #ddd"
        },
        sort_action="native",
        filter_action="native",
        page_size=100,
    )
    return table