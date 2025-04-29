# vol_table.py
# This Dash page displays the Vol Table with refined UI and drill‑down details.
# We rely on a global ticker from the store; the summary will load as soon as you navigate here.

import dash
import dash_bootstrap_components as dbc
from dash import dcc, html, dash_table, Input, Output, State, callback, ctx
import pandas as pd
from datetime import datetime, date
import requests
from io import StringIO

# Register this script as a Dash page with path="/vol_table"
dash.register_page(__name__, path="/vol_table", name="Vol Table")


# ------------------------------------------------
# Helper Functions
# ------------------------------------------------

def format_date(yyyymmdd):
    """Converts a YYYYMMDD string to a datetime object."""
    try:
        return datetime.strptime(str(yyyymmdd), '%Y%m%d')
    except ValueError:
        return None


def list_expirations(root):
    resp = requests.get(
        "http://127.0.0.1:25510/v2/list/expirations",
        params={'root': root},
        timeout=10
    )
    resp.raise_for_status()
    data = resp.json()
    if data.get('header', {}).get('error_type'):
        raise Exception(data['header'].get('error_msg', 'Unknown error'))
    exps = data.get('response', [])
    if not exps:
        raise Exception(f"No expirations for '{root}'")
    return exps


def fetch_option_chain(root, expiration):
    resp = requests.get(
        "http://127.0.0.1:25510/v2/bulk_snapshot/option/greeks",
        params={'root': root, 'exp': expiration, 'use_csv': 'true'},
        timeout=10
    )
    resp.raise_for_status()
    df = pd.read_csv(StringIO(resp.text))
    if df.empty:
        raise Exception(f"No option data for {root} @ {expiration}")
    return df


def detect_scaling_factor(df):
    for factor in (1, 10, 100, 1000, 10000):
        if (df['strike'] / factor).between(1, 10000).all():
            return factor
    raise Exception("Can't detect strike scaling")


def build_option_chain_table(df_raw):
    try:
        scale = detect_scaling_factor(df_raw)
    except:
        scale = 1
    df_raw['strike'] = df_raw['strike'].astype(float) / scale

    calls = df_raw[df_raw['right'] == 'C'].copy()
    puts  = df_raw[df_raw['right'] == 'P'].copy()

    calls.rename(columns={
        'bid': 'Call Bid', 'ask': 'Call Ask',
        'implied_vol': 'IV Call', 'delta': 'Delta Call',
        'theta': 'Theta Call', 'vega': 'Vega Call', 'rho': 'Rho Call'
    }, inplace=True)
    puts.rename(columns={
        'bid': 'Put Bid', 'ask': 'Put Ask',
        'implied_vol': 'IV Put', 'delta': 'Delta Put',
        'theta': 'Theta Put', 'vega': 'Vega Put', 'rho': 'Rho Put'
    }, inplace=True)

    calls = calls[['strike','Call Bid','Call Ask','IV Call','Delta Call','Theta Call','Vega Call','Rho Call']]
    puts  = puts[['strike','Put Bid','Put Ask','IV Put','Delta Put','Theta Put','Vega Put','Rho Put']]

    merged = pd.merge(calls, puts, on='strike', how='outer')
    merged.rename(columns={'strike': 'Strike'}, inplace=True)
    merged['Strike'] = (merged['Strike'] / 10).round(0).astype(int)

    cols = [
        'Rho Call','Vega Call','Theta Call','Delta Call','IV Call',
        'Call Bid','Call Ask','Strike',
        'Put Bid','Put Ask','IV Put','Delta Put','Theta Put','Vega Put','Rho Put'
    ]
    return merged.reindex(columns=cols).sort_values('Strike')


def fetch_stock_eod(ticker, start_date, end_date):
    resp = requests.get(
        "http://127.0.0.1:25510/v2/hist/stock/eod",
        params={
            "root": ticker.upper(),
            "start_date": start_date,
            "end_date": end_date,
            "use_csv": "true"
        },
        timeout=10
    )
    resp.raise_for_status()
    return pd.read_csv(StringIO(resp.text))


def get_stock_price(ticker):
    end = datetime.today().strftime("%Y%m%d")
    start = (datetime.today() - pd.Timedelta(days=500)).strftime("%Y%m%d")
    df = fetch_stock_eod(ticker, start, end)
    df['date'] = pd.to_datetime(df['date'], format='%Y%m%d')
    return df.sort_values('date').iloc[-1]['close']


def get_bid_ask_for_target(df, target, delta_col, bid_col, ask_col):
    """
    Find the row whose <delta_col> is closest to <target>,
    and return "bid x ask".
    """
    idx = (df[delta_col] - target).abs().idxmin()
    bid = df.loc[idx, bid_col]
    ask = df.loc[idx, ask_col]
    return f"{bid:.2f} x {ask:.2f}"


def get_midprice(df, target, delta_col, bid_col, ask_col):
    """
    Find the row whose <delta_col> is closest to <target>,
    and return the mid-price.
    """
    idx = (df[delta_col] - target).abs().idxmin()
    bid = df.loc[idx, bid_col]
    ask = df.loc[idx, ask_col]
    return f"{(bid + ask) / 2:.2f}"


# ------------------------------------------------
# Layout
# ------------------------------------------------
layout = dbc.Container(
    fluid=True,
    style={"backgroundColor":"#f5f5f5","minHeight":"100vh","padding":"20px"},
    children=[
        dbc.Alert(
            id="vol-table-ticker-display",
            color="info",
            className="text-center mb-3",
            children=""
        ),
        dbc.Card(
            dbc.CardBody([
                html.H4("Vol Table", className="text-center mb-3"),
                html.P(
                    "This page fetches the option chain summary for the globally selected ticker.",
                    className="text-center mb-4"
                ),
                dbc.Row(
                    dbc.Col(
                        dcc.RadioItems(
                            id="display-option-radio",
                            options=[
                                {"label":"bid x ask","value":"bidask"},
                                {"label":"mid-price","value":"midprice"},
                            ],
                            value="bidask",
                            labelStyle={"display":"inline-block","marginRight":"20px"}
                        ),
                        width={"size":6,"offset":3}
                    ),
                    className="mb-3"
                ),
                dcc.Loading(
                    dash_table.DataTable(
                        id="summary-table",
                        columns=[],
                        data=[],
                        merge_duplicate_headers=True,
                        fixed_rows={"headers":True},
                        style_table={'overflowY':'auto','maxHeight':'500px'},
                        style_cell={'textAlign':'center','padding':'8px'},
                        style_header={'backgroundColor':"#e9ecef",'fontWeight':'bold','border':'none'},
                        page_action='none',
                        row_selectable='single'
                    ),
                    type="default"
                ),
            ]),
            className="mt-4 shadow",
            style={"borderRadius":"10px","padding":"20px"}
        ),
        dbc.Modal(
            [
                dbc.ModalHeader("Detailed Option Chain"),
                dbc.ModalBody(id="modal-body-detail", style={"maxHeight":"600px","overflowY":"auto"}),
                dbc.ModalFooter(dbc.Button("Close", id="close-modal", className="ml-auto"))
            ],
            id="detail-modal", centered=True, is_open=False, size="xl"
        )
    ]
)


# ------------------------------------------------
# Callback: Update Summary Table on page load or interaction
# ------------------------------------------------
@callback(
    Output("vol-table-ticker-display", "children"),
    Output("summary-table", "data"),
    Output("summary-table", "columns"),
    Input("display-option-radio", "value"),
    Input("global-ticker-store", "data"),
)
def update_summary(display_option, global_ticker):
    # If no ticker, show nothing
    if not global_ticker:
        return "", [], []

    ticker = global_ticker.upper()

    # Fetch expirations
    try:
        exps = list_expirations(ticker)
    except Exception as e:
        return f"Error fetching expirations: {e}", [], []

    today = date.today()
    future = [e for e in exps if format_date(e) and format_date(e).date() >= today]
    future.sort(key=lambda x: format_date(x))

    call_map = {"Calls ATM":0.50,"Calls 15D":0.15,"Calls 25D":0.25,"Calls 35D":0.35}
    put_map  = {"Puts ATM":-0.50,"Puts 15D":-0.15,"Puts 25D":-0.25,"Puts 35D":-0.35}

    rows = []
    for exp in future:
        exp_dt = format_date(exp)
        dte = (exp_dt.date() - today).days
        try:
            df = build_option_chain_table(fetch_option_chain(ticker, exp))
        except:
            continue

        row = {"Expiration": exp_dt.strftime("%d/%m/%Y"), "DTE": dte}
        try:
            idx = (df["Delta Call"] - 0.50).abs().idxmin()
            move = (df.loc[idx, "Call Bid"] + df.loc[idx, "Put Bid"]) / df.loc[idx, "Strike"]
            row["Implied Move"] = f"{move*100:.2f}%"
        except:
            row["Implied Move"] = "N/A"

        for name, target in call_map.items():
            fn = get_bid_ask_for_target if display_option == "bidask" else get_midprice
            try:
                row[name] = fn(df, target, "Delta Call", "Call Bid", "Call Ask")
            except:
                row[name] = "N/A"

        for name, target in put_map.items():
            fn = get_bid_ask_for_target if display_option == "bidask" else get_midprice
            try:
                row[name] = fn(df, target, "Delta Put", "Put Bid", "Put Ask")
            except:
                row[name] = "N/A"

        row["raw_exp"], row["ticker"] = exp, ticker
        rows.append(row)

    if not rows:
        return "", [], []

    columns = (
        [{"name":"Expiration","id":"Expiration"},{"name":"DTE","id":"DTE"},{"name":"Implied Move","id":"Implied Move"}]
        + [{"name":k,"id":k} for k in call_map]
        + [{"name":k,"id":k} for k in put_map]
    )

    return "", rows, columns


# ------------------------------------------------
# Callback: Drill‑Down Modal
# ------------------------------------------------
@callback(
    Output("detail-modal", "is_open"),
    Output("modal-body-detail", "children"),
    Input("summary-table", "active_cell"),
    Input("close-modal", "n_clicks"),
    State("summary-table", "data"),
    State("detail-modal", "is_open"),
)
def toggle_modal(active_cell, close_clicks, data, is_open):
    triggered = ctx.triggered_id or ""
    if triggered == "close-modal":
        return False, ""
    if active_cell and data:
        idx = active_cell["row"]
        row = data[idx]
        exp, tick = row["raw_exp"], row["ticker"]
        try:
            df = build_option_chain_table(fetch_option_chain(tick, exp))
            detail = dash_table.DataTable(
                columns=[{"name":c,"id":c} for c in df.columns],
                data=df.to_dict("records"),
                fixed_rows={"headers":True},
                style_table={"overflowY":"auto","maxHeight":"400px"},
                style_cell={"textAlign":"center","padding":"8px"},
                style_header={"backgroundColor":"#e9ecef","fontWeight":"bold"},
                page_action="none"
            )
        except Exception as e:
            detail = f"Error loading details: {e}"
        return True, detail
    return is_open, ""
