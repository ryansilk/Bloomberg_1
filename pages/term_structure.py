# term_structure.py
"""
Dash page for analyzing Option Term Structure:
 - Allows selecting multiple expirations,
 - Compares any chosen columns (Greeks, bids/asks, etc.) across those expirations,
 - Provides chart and table views.
Relies on a global ticker from dcc.Store.
"""

import dash
import dash_bootstrap_components as dbc
import requests
from datetime import datetime
import pandas as pd
from dash import dcc, html, dash_table, callback, Input, Output
from io import StringIO
import plotly.graph_objects as go

##############################################################################
# 1. Register Page
##############################################################################
dash.register_page(__name__, path="/term_structure", title="Term Structure")


##############################################################################
# 2. Helper Functions
##############################################################################

def format_date(yyyymmdd):
    """Convert YYYYMMDD string to datetime, or return None on error."""
    try:
        return datetime.strptime(str(yyyymmdd), "%Y%m%d")
    except ValueError:
        return None


def list_expirations(root):
    """Fetch available option expirations for `root`."""
    url = "http://127.0.0.1:25510/v2/list/expirations"
    resp = requests.get(url, params={"root": root}, timeout=10)
    resp.raise_for_status()
    data = resp.json()
    if data.get("header", {}).get("error_type"):
        raise Exception(data["header"].get("error_msg", "Unknown error"))
    exps = data.get("response", [])
    if not exps:
        raise Exception(f"No expirations found for '{root}'")
    return exps


def fetch_option_chain(root, expiration):
    """Fetch raw option-greeks CSV for a given root and expiration."""
    url = "http://127.0.0.1:25510/v2/bulk_snapshot/option/greeks"
    resp = requests.get(
        url,
        params={"root": root, "exp": expiration, "use_csv": "true"},
        timeout=10
    )
    resp.raise_for_status()
    df = pd.read_csv(StringIO(resp.text))
    if df.empty:
        raise Exception(f"No data for {root} exp {expiration}")
    return df


def detect_scaling_factor(df):
    """Identify correct divisor for the 'strike' column."""
    for factor in (1, 10, 100, 1000, 10000):
        scaled = df["strike"] / factor
        if scaled.between(1, 10000).all():
            return factor
    raise Exception("Cannot detect strike scaling")


def build_option_chain_table(df_raw):
    """Return merged calls/puts DataFrame with normalized strike prices."""
    try:
        factor = detect_scaling_factor(df_raw)
    except Exception:
        factor = 1
    df_raw["strike"] = df_raw["strike"].astype(float) / factor

    calls = df_raw[df_raw["right"] == "C"].copy()
    puts  = df_raw[df_raw["right"] == "P"].copy()

    calls.rename(columns={
        "bid": "Call Bid", "ask": "Call Ask",
        "delta": "Delta Call", "theta": "Theta Call",
        "vega": "Vega Call", "rho": "Rho Call"
    }, inplace=True)
    puts.rename(columns={
        "bid": "Put Bid", "ask": "Put Ask",
        "delta": "Delta Put", "theta": "Theta Put",
        "vega": "Vega Put", "rho": "Rho Put"
    }, inplace=True)

    calls = calls[["strike", "Call Bid", "Call Ask",
                   "Delta Call", "Theta Call", "Vega Call", "Rho Call"]]
    puts  = puts[["strike", "Put Bid", "Put Ask",
                  "Delta Put", "Theta Put", "Vega Put", "Rho Put"]]

    merged = pd.merge(calls, puts, on="strike", how="outer")
    merged.rename(columns={"strike": "Strike"}, inplace=True)
    merged["Strike"] = (merged["Strike"] / 10).round(0).astype(int)

    cols = [
        "Rho Call", "Vega Call", "Theta Call", "Delta Call",
        "Call Bid", "Call Ask", "Strike",
        "Put Bid", "Put Ask", "Delta Put", "Theta Put", "Vega Put", "Rho Put"
    ]
    merged = merged.reindex(columns=cols).sort_values("Strike")
    return merged


# Simple cache to avoid refetching the same data repeatedly
OPTION_CHAIN_CACHE = {}
def get_cached_option_chain(root, exp):
    key = (root, exp)
    if key not in OPTION_CHAIN_CACHE:
        OPTION_CHAIN_CACHE[key] = fetch_option_chain(root, exp)
    return OPTION_CHAIN_CACHE[key]


##############################################################################
# 3. Page Layout (Term Structure)
##############################################################################
layout = dbc.Container(
    [
        # Expiration selection & column selector
        dbc.Row(
            [
                dbc.Col([
                    dbc.Label("Select Expirations:"),
                    dcc.Dropdown(
                        id="expiration-comparison-dropdown",
                        multi=True,
                        placeholder="e.g., 2025-05-16"
                    )
                ], md=6),
                dbc.Col([
                    dbc.Label("Select Columns:"),
                    dcc.Dropdown(
                        id="greek-selector-dropdown",
                        options=[
                            {"label": col, "value": col}
                            for col in [
                                "Rho Call", "Vega Call", "Theta Call", "Delta Call",
                                "Call Bid", "Call Ask", "Strike",
                                "Put Bid", "Put Ask", "Delta Put", "Theta Put", "Vega Put", "Rho Put"
                            ]
                        ],
                        value=["Delta Call"],
                        multi=True
                    )
                ], md=6),
            ],
            className="mb-3"
        ),

        # Chart type radio
        dbc.Row(
            dbc.Col([
                dbc.Label("Chart Type:"),
                dcc.RadioItems(
                    id="chart-type-radio",
                    options=[
                        {"label": "Line", "value": "line"},
                        {"label": "Scatter", "value": "scatter"}
                    ],
                    value="line",
                    inline=True
                )
            ], md=6),
            className="mb-3"
        ),

        # Comparison view
        dbc.Tabs(
            [
                dbc.Tab(
                    dcc.Loading(dcc.Graph(id="greek-comparison-chart"), type="default"),
                    label="Chart View", tab_id="chart_view"
                ),
                dbc.Tab(
                    dcc.Loading(
                        dash_table.DataTable(
                            id="greek-data-table",
                            columns=[], data=[],
                            fixed_rows={"headers": True},
                            style_table={"maxHeight": "400px", "overflowY": "auto"},
                            style_cell={"textAlign": "center", "padding": "10px"},
                            style_header={
                                "backgroundColor": "#f8f9fa",
                                "fontWeight": "bold"
                            },
                            style_data_conditional=[
                                {"if": {"row_index": "odd"}, "backgroundColor": "#f2f2f2"}
                            ],
                            sort_action="native",
                            filter_action="native",
                            column_selectable="multi",
                            page_size=20
                        ),
                        type="default"
                    ),
                    label="Data Table", tab_id="data_table"
                )
            ],
            id="greek-tabs",
            active_tab="chart_view"
        ),
    ],
    fluid=True,
    className="pt-4"
)


##############################################################################
# 4. Callbacks (Term Structure)
##############################################################################

@callback(
    Output("expiration-comparison-dropdown", "options"),
    Input("global-ticker-store", "data")
)
def update_expiration_options(ticker):
    if not ticker:
        return []
    exps = list_expirations(ticker.upper())
    today = datetime.today().date()
    future = [e for e in exps if format_date(e) and format_date(e).date() >= today]
    future.sort()
    return [
        {"label": format_date(e).strftime("%Y-%m-%d"), "value": e}
        for e in future
    ]


@callback(
    Output("greek-comparison-chart", "figure"),
    Input("global-ticker-store", "data"),
    Input("expiration-comparison-dropdown", "value"),
    Input("greek-selector-dropdown", "value"),
    Input("chart-type-radio", "value"),
)
def update_greek_comparison_chart(ticker, exps, cols, chart_type):
    fig = go.Figure()
    if not ticker or not exps or not cols:
        fig.add_annotation(
            text="Select a ticker, expirations, and columns",
            xref="paper", yref="paper", showarrow=False
        )
        return fig

    root = ticker.upper()
    exps = exps if isinstance(exps, list) else [exps]
    cols = cols if isinstance(cols, list) else [cols]

    for exp in exps:
        try:
            df = build_option_chain_table(get_cached_option_chain(root, exp))
            label = format_date(exp).strftime("%Y-%m-%d")
            for c in cols:
                if c in df.columns:
                    mode = "lines+markers" if chart_type == "line" else "markers"
                    fig.add_trace(go.Scatter(
                        x=df["Strike"], y=df[c], mode=mode,
                        name=f"{c} ({label})"
                    ))
        except Exception as e:
            fig.add_annotation(
                text=f"Error for {exp}: {e}",
                xref="paper", yref="paper", showarrow=False,
                font={"color": "red"}
            )

    fig.update_layout(template="plotly_white", hovermode="x unified")
    return fig


@callback(
    Output("greek-data-table", "data"),
    Output("greek-data-table", "columns"),
    Input("global-ticker-store", "data"),
    Input("expiration-comparison-dropdown", "value"),
    Input("greek-selector-dropdown", "value"),
)
def update_greek_data_table(ticker, exps, cols):
    if not ticker or not exps or not cols:
        return [], []
    root = ticker.upper()
    exps = exps if isinstance(exps, list) else [exps]
    cols = cols if isinstance(cols, list) else [cols]

    pieces = []
    for exp in exps:
        try:
            df = build_option_chain_table(get_cached_option_chain(root, exp))
            df["Expiration"] = format_date(exp).strftime("%Y-%m-%d")
            keep = ["Expiration", "Strike"] + [c for c in cols if c in df.columns]
            pieces.append(df[keep])
        except Exception:
            continue

    if not pieces:
        return [], []

    final = pd.concat(pieces, ignore_index=True).sort_values(["Expiration", "Strike"])
    columns = [{"name": c, "id": c} for c in final.columns]
    return final.to_dict("records"), columns