# vol_smile.py
"""
Dash page for visualizing the Volatility Smile (IV Call & IV Put):
 - Allows selecting multiple expirations,
 - Compares IV Call and IV Put across those expirations,
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
dash.register_page(__name__, path="/vol_smile", title="Vol Smile")


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
    """Return merged calls/puts DataFrame with normalized strike prices and IV columns."""
    # scale strikes
    try:
        factor = detect_scaling_factor(df_raw)
    except Exception:
        factor = 1
    df_raw["strike"] = df_raw["strike"].astype(float) / factor

    # separate calls and puts
    calls = df_raw[df_raw["right"] == "C"].copy()
    puts  = df_raw[df_raw["right"] == "P"].copy()

    # keep and rename only IV columns plus strike
    calls = calls[["strike", "implied_vol"]].rename(columns={"implied_vol": "IV Call"})
    puts  = puts[["strike", "implied_vol"]].rename(columns={"implied_vol": "IV Put"})

    # merge on strike
    merged = pd.merge(calls, puts, on="strike", how="outer")
    merged.rename(columns={"strike": "Strike"}, inplace=True)
    # Strike display: divide by 10 and round if needed
    merged["Strike"] = (merged["Strike"] / 10).round(0).astype(int)

    # ensure column order
    merged = merged[["Strike", "IV Call", "IV Put"]].sort_values("Strike")
    return merged


# cache to avoid refetching same expiration
OPTION_CHAIN_CACHE = {}
def get_cached_option_chain(root, exp):
    key = (root, exp)
    if key not in OPTION_CHAIN_CACHE:
        OPTION_CHAIN_CACHE[key] = fetch_option_chain(root, exp)
    return OPTION_CHAIN_CACHE[key]


##############################################################################
# 3. Page Layout (Vol Smile)
##############################################################################
layout = dbc.Container(
    [
        dbc.Row(
            [
                dbc.Col([
                    dbc.Label("Select Expirations:"),
                    dcc.Dropdown(
                        id="expiration-smile-dropdown",
                        multi=True,
                        placeholder="e.g., 2025-05-16"
                    ),
                ], md=6),
                dbc.Col([
                    dbc.Label("Columns:"),
                    dcc.Dropdown(
                        id="smile-column-dropdown",
                        options=[
                            {"label": "IV Call", "value": "IV Call"},
                            {"label": "IV Put",  "value": "IV Put"},
                        ],
                        value=["IV Call", "IV Put"],
                        multi=True,
                        clearable=False
                    ),
                ], md=6),
            ],
            className="mb-3"
        ),
        dbc.Row(
            dbc.Col([
                dbc.Label("Chart Type:"),
                dcc.RadioItems(
                    id="chart-type-smile-radio",
                    options=[
                        {"label": "Line",    "value": "line"},
                        {"label": "Scatter", "value": "scatter"},
                    ],
                    value="line",
                    inline=True
                )
            ], md=6),
            className="mb-3"
        ),
        dbc.Tabs(
            [
                dbc.Tab(
                    dcc.Loading(dcc.Graph(id="vol-smile-chart"), type="default"),
                    label="Chart View", tab_id="chart_view"
                ),
                dbc.Tab(
                    dcc.Loading(
                        dash_table.DataTable(
                            id="vol-smile-data-table",
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
                            page_size=20,
                            export_format="csv"
                        ),
                        type="default"
                    ),
                    label="Data Table", tab_id="data_table"
                )
            ],
            id="vol-smile-tabs",
            active_tab="chart_view"
        ),
    ],
    fluid=True,
    className="pt-4"
)


##############################################################################
# 4. Callbacks (Vol Smile)
##############################################################################

@callback(
    Output("expiration-smile-dropdown", "options"),
    Input("global-ticker-store", "data")
)
def update_expirations(global_ticker):
    if not global_ticker:
        return []
    exps = list_expirations(global_ticker.upper())
    today = datetime.today().date()
    future = [e for e in exps if format_date(e) and format_date(e).date() >= today]
    future.sort()
    return [
        {"label": format_date(e).strftime("%Y-%m-%d"), "value": e}
        for e in future
    ]


@callback(
    Output("vol-smile-chart", "figure"),
    Input("global-ticker-store", "data"),
    Input("expiration-smile-dropdown", "value"),
    Input("smile-column-dropdown", "value"),
    Input("chart-type-smile-radio", "value"),
)
def update_vol_smile_chart(global_ticker, expirations, columns, chart_type):
    fig = go.Figure()
    if not global_ticker or not expirations or not columns:
        fig.add_annotation(text="Select a ticker, expirations & columns",
                           xref="paper", yref="paper", showarrow=False)
        return fig

    root = global_ticker.upper()
    exps = expirations if isinstance(expirations, list) else [expirations]
    cols = columns if isinstance(columns, list) else [columns]

    for exp in exps:
        try:
            raw = get_cached_option_chain(root, exp)
            df = build_option_chain_table(raw)
            label = format_date(exp).strftime("%Y-%m-%d")
            for col in cols:
                if col in df.columns:
                    mode = "lines+markers" if chart_type == "line" else "markers"
                    fig.add_trace(go.Scatter(
                        x=df["Strike"], y=df[col], mode=mode,
                        name=f"{col} ({label})"
                    ))
        except Exception as e:
            fig.add_annotation(text=f"Error for {exp}: {e}",
                               xref="paper", yref="paper", showarrow=False,
                               font={"color": "red"})

    fig.update_layout(template="plotly_white", hovermode="x unified",
                      xaxis_title="Strike", yaxis_title="Implied Volatility")
    return fig


@callback(
    Output("vol-smile-data-table", "data"),
    Output("vol-smile-data-table", "columns"),
    Input("global-ticker-store", "data"),
    Input("expiration-smile-dropdown", "value"),
    Input("smile-column-dropdown", "value"),
)
def update_vol_smile_table(global_ticker, expirations, columns):
    if not global_ticker or not expirations or not columns:
        return [], []

    root = global_ticker.upper()
    exps = expirations if isinstance(expirations, list) else [expirations]
    cols = columns if isinstance(columns, list) else [columns]

    rows = []
    for exp in exps:
        try:
            raw = get_cached_option_chain(root, exp)
            df = build_option_chain_table(raw)
            df["Expiration"] = format_date(exp).strftime("%Y-%m-%d")
            keep = ["Expiration", "Strike"] + [c for c in cols if c in df.columns]
            rows.append(df[keep])
        except Exception:
            continue

    if not rows:
        return [], []

    final = pd.concat(rows, ignore_index=True).sort_values(["Expiration", "Strike"])
    cols_out = [{"name": c, "id": c} for c in final.columns]
    return final.to_dict("records"), cols_out