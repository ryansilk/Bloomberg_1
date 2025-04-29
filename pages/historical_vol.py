# historical_vol.py

import dash
import dash_bootstrap_components as dbc
from dash import dcc, html, dash_table, Input, Output, callback, no_update
import requests
import pandas as pd
import numpy as np
from datetime import datetime, date, timedelta
import plotly.express as px
import plotly.graph_objects as go

# Register the page
dash.register_page(
    __name__,
    path="/historical_vol",
    title="Historical Vol",
    name="Historical Vol"
)

# Only these HV windows
HV_WINDOWS = {
    "hv_5": 5,
    "hv_10": 10,
    "hv_20": 20,
    "hv_50": 50,
    "hv_100": 100
}


def fetch_compute_hv(symbol, start_api, end_api):
    """
    Fetches EOD data and computes HV windows for a symbol.
    Returns DataFrame with Date, Close, hv_5, ..., hv_100.
    """
    url = "http://127.0.0.1:25510/v2/hist/stock/eod"
    params = {
        "root": symbol.upper(),
        "start_date": start_api,
        "end_date": end_api,
        "use_csv": "false"
    }
    resp = requests.get(url, params=params, timeout=10)
    resp.raise_for_status()
    data = resp.json()
    if not data.get("response"):
        return None

    df = pd.DataFrame(data["response"], columns=data["header"]["format"])
    df.rename(columns={"date": "Date", "close": "Close"}, inplace=True)
    df["Date"] = pd.to_datetime(df["Date"], format="%Y%m%d")
    df.sort_values("Date", inplace=True)
    df["log_ret"] = np.log(df["Close"] / df["Close"].shift(1))

    for key, window in HV_WINDOWS.items():
        df[key] = df["log_ret"].rolling(window).std() * np.sqrt(252)

    return df


layout = dbc.Container(fluid=True, children=[

    # Filters row: date range, compare ticker, chart type, HV windows
    dbc.Row([
        dbc.Col([
            html.Label("Date Range"),
            dcc.DatePickerRange(
                id="historical-vol-date-range",
                start_date=(date.today() - timedelta(days=1) - timedelta(days=180)),
                end_date=(date.today() - timedelta(days=1)),
                display_format="YYYY-MM-DD",
                updatemode="bothdates"
            )
        ], md=3),
        dbc.Col([
            html.Label("Compare Symbols"),
            dcc.Input(
                id="historical-vol-compare-input",
                placeholder="e.g., MSFT, GOOGL",
                debounce=True,
                style={"width": "100%"}
            )
        ], md=3),
        dbc.Col([
            html.Label("Chart Type"),
            dcc.RadioItems(
                id="historical-vol-chart-type",
                options=[
                    {"label": "Line", "value": "Line"},
                    {"label": "Histogram", "value": "Histogram"},
                    {"label": "Box Plot", "value": "Box"}
                ],
                value="Line",
                inline=True
            )
        ], md=3),
        dbc.Col([
            html.Label("HV Windows"),
            dcc.Checklist(
                id="historical-vol-hv-checklist",
                options=[{"label": f"HV({w})", "value": f"hv_{w}"} for w in [5,10,20,50,100]],
                value=["hv_5", "hv_10", "hv_20", "hv_50", "hv_100"],
                inline=True
            )
        ], md=3),
    ], className="mb-4"),

    # Summary cards
    html.Div(id="historical-vol-summary-cards", className="mb-4"),

    # Full-width chart
    dbc.Card(
        dbc.CardBody(
            dcc.Loading(dcc.Graph(id="historical-vol-hv-graph"), type="default")
        ),
        className="mb-4 shadow-sm"
    ),

    # Single table with all historical data
    dbc.Card(
        dbc.CardBody([
            html.H4("Historical Data", className="card-title"),
            dash_table.DataTable(
                id="historical-vol-full-table",
                columns=[],
                data=[],
                page_size=20,
                virtualization=True,
                fixed_rows={"headers": True},
                style_table={"overflowX": "auto"},
                style_header={"backgroundColor": "#f1f1f1", "fontWeight": "bold"},
                style_cell={"textAlign": "center", "padding": "6px"},
            )
        ]),
        className="mb-4 shadow-sm"
    )
])


@callback(
    Output("historical-vol-summary-cards", "children"),
    Output("historical-vol-hv-graph", "figure"),
    Output("historical-vol-full-table", "columns"),
    Output("historical-vol-full-table", "data"),
    Input("global-ticker-store", "data"),
    Input("historical-vol-date-range", "start_date"),
    Input("historical-vol-date-range", "end_date"),
    Input("historical-vol-compare-input", "value"),
    Input("historical-vol-hv-checklist", "value"),
    Input("historical-vol-chart-type", "value"),
)
def update_dashboard(global_ticker, start_date, end_date, compare_input, hv_selected, chart_type):
    # require ticker
    if not global_ticker:
        return no_update, no_update, [], []

    # lookup period ends yesterday
    if end_date:
        end_api = datetime.strptime(end_date[:10], "%Y-%m-%d").strftime("%Y%m%d")
    else:
        end_api = (date.today() - timedelta(days=1)).strftime("%Y%m%d")
    if start_date:
        start_api = datetime.strptime(start_date[:10], "%Y-%m-%d").strftime("%Y%m%d")
    else:
        start_api = (date.today() - timedelta(days=1) - timedelta(days=180)).strftime("%Y%m%d")

    main_sym = global_ticker.upper()
    symbols = [main_sym]
    if compare_input:
        extras = [s.strip().upper() for s in compare_input.split(",") if s.strip()]
        symbols += extras

    # fetch & compute HV
    data_dict = {}
    for sym in symbols:
        df_sym = fetch_compute_hv(sym, start_api, end_api)
        if df_sym is not None:
            data_dict[sym] = df_sym

    if main_sym not in data_dict:
        alert = dbc.Alert(f"Could not fetch data for {main_sym}.", color="danger")
        return [alert], go.Figure(), [], []

    # summary cards
    df_main = data_dict[main_sym]
    df_valid = df_main.dropna(subset=hv_selected)
    if df_valid.empty:
        alert = dbc.Alert(f"Insufficient data for {main_sym}.", color="warning")
        return [alert], go.Figure(), [], []
    latest = df_valid.iloc[-1]

    cards = []
    for hv_key in hv_selected:
        window = HV_WINDOWS[hv_key]
        val = latest[hv_key] * 100
        cards.append(
            dbc.Col(
                dbc.Card(
                    [dbc.CardHeader(f"HV({window})"), dbc.CardBody(f"{val:.2f}%")],
                    className="text-center"
                ),
                width="auto"
            )
        )
    summary = dbc.Row(cards, className="gx-2")

    # build figure
    fig = go.Figure()
    if chart_type == "Line":
        for sym, df_sym in data_dict.items():
            for hv_key in hv_selected:
                fig.add_trace(go.Scatter(
                    x=df_sym["Date"], y=df_sym[hv_key],
                    mode="lines", name=f"{sym} HV({HV_WINDOWS[hv_key]})"
                ))
        fig.update_layout(
            yaxis_tickformat=",.0%",
            template="plotly_white",
            hovermode="x unified"
        )
    elif chart_type == "Histogram":
        frames = []
        for sym, df_sym in data_dict.items():
            for hv_key in hv_selected:
                tmp = df_sym[["Date", hv_key]].dropna().copy()
                tmp["Window"] = f"HV({HV_WINDOWS[hv_key]})"
                tmp["Symbol"] = sym
                tmp.rename(columns={hv_key: "HV"}, inplace=True)
                frames.append(tmp)
        combined = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
        fig = px.histogram(
            combined, x="HV", color="Symbol", facet_col="Window",
            nbins=50, template="plotly_white"
        )
        fig.update_layout(yaxis_title="Count")
    else:  # Box
        frames = []
        for sym, df_sym in data_dict.items():
            for hv_key in hv_selected:
                tmp = df_sym[["Date", hv_key]].dropna().copy()
                tmp["Window"] = f"HV({HV_WINDOWS[hv_key]})"
                tmp["Symbol"] = sym
                tmp.rename(columns={hv_key: "HV"}, inplace=True)
                frames.append(tmp)
        combined = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
        fig = px.box(
            combined, x="Window", y="HV", color="Symbol",
            template="plotly_white"
        )
        fig.update_layout(yaxis_tickformat=",.0%")

    # full table
    full = df_main.dropna(subset=hv_selected).copy()
    full["DateStr"] = full["Date"].dt.strftime("%Y-%m-%d")
    columns = (
        [{"name": "Date", "id": "DateStr"},
         {"name": "Close", "id": "Close", "type": "numeric", "format": {"specifier": ".2f"}}]
        + [{"name": f"HV({HV_WINDOWS[k]})", "id": k, "type": "numeric", "format": {"specifier": ".2f"}}
           for k in hv_selected]
    )
    data = full.to_dict("records")

    return summary, fig, columns, data
