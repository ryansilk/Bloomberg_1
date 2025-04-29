# pages/option_chain.py

import dash
import dash_bootstrap_components as dbc
import requests
from datetime import datetime
import pandas as pd
from dash import dcc, html, dash_table, callback, Input, Output
from io import StringIO
import plotly.graph_objects as go

dash.register_page(__name__, path="/other_page", title="Option Chain")

BASE_URL = "http://127.0.0.1:25510/v2"


def format_date(yyyymmdd):
    try:
        return datetime.strptime(str(yyyymmdd), "%Y%m%d")
    except ValueError:
        return None


def list_expirations(root):
    resp = requests.get(f"{BASE_URL}/list/expirations", params={"root": root}, timeout=10)
    resp.raise_for_status()
    js = resp.json()
    if js["header"].get("error_type"):
        raise Exception(js["header"]["error_msg"])
    return js["response"] or []


def fetch_option_chain(root, expiration):
    resp = requests.get(
        f"{BASE_URL}/bulk_snapshot/option/greeks",
        params={"root": root, "exp": expiration, "use_csv": "true"},
        timeout=10,
    )
    resp.raise_for_status()
    df = pd.read_csv(StringIO(resp.text))
    if df.empty:
        raise Exception(f"No option data for {root}@{expiration}")
    return df


def detect_scaling_factor(df):
    for f in (1, 10, 100, 1000, 10000):
        if (df["strike"] / f).between(1, 10000).all():
            return f
    return 10  # fallback


def build_option_chain_table(df_raw):
    # always scale by 10
    scale = detect_scaling_factor(df_raw)
    df = df_raw.copy()
    df["strike"] = df["strike"].astype(float) / scale

    calls = df[df["right"] == "C"].copy()
    puts = df[df["right"] == "P"].copy()

    calls.rename(
        columns={
            "bid": "Call Bid",
            "ask": "Call Ask",
            "implied_vol": "IV Call",
            "delta": "Delta Call",
            "theta": "Theta Call",
            "vega": "Vega Call",
            "rho": "Rho Call",
        },
        inplace=True,
    )
    puts.rename(
        columns={
            "bid": "Put Bid",
            "ask": "Put Ask",
            "implied_vol": "IV Put",
            "delta": "Delta Put",
            "theta": "Theta Put",
            "vega": "Vega Put",
            "rho": "Rho Put",
        },
        inplace=True,
    )

    calls = calls[["strike", "Rho Call", "Vega Call", "Theta Call", "Delta Call", "IV Call", "Call Bid", "Call Ask"]]
    puts = puts[["strike", "Put Bid", "Put Ask", "IV Put", "Delta Put", "Theta Put", "Vega Put", "Rho Put"]]

    merged = pd.merge(calls, puts, on="strike", how="outer").rename(columns={"strike": "Strike"})
    merged["Strike"] = merged["Strike"].round(2)

    cols = [
        "Rho Call",
        "Vega Call",
        "Theta Call",
        "Delta Call",
        "IV Call",
        "Call Bid",
        "Call Ask",
        "Strike",
        "Put Bid",
        "Put Ask",
        "IV Put",
        "Delta Put",
        "Theta Put",
        "Vega Put",
        "Rho Put",
    ]
    return merged.reindex(columns=cols).sort_values("Strike")


def build_heatmap(df):
    greek_cols = [
        "Delta Call",
        "Theta Call",
        "Vega Call",
        "Rho Call",
        "Delta Put",
        "Theta Put",
        "Vega Put",
        "Rho Put",
    ]
    melt = df[["Strike"] + greek_cols].melt(
        id_vars="Strike", var_name="Greek", value_name="Value"
    )
    pivot = melt.pivot(index="Greek", columns="Strike", values="Value")
    fig = go.Figure(
        go.Heatmap(
            z=pivot.values,
            x=[str(x) for x in pivot.columns],
            y=pivot.index,
            colorscale="Viridis",
            colorbar=dict(title="Value"),
        )
    )
    fig.update_layout(
        title="Greeks Heatmap",
        xaxis_title="Strike",
        yaxis_title="Greek",
        template="plotly_white",
        margin=dict(l=50, r=50, t=50, b=50),
    )
    return fig


layout = dbc.Container(
    [
        dbc.Accordion(
            id="otherpage-option-accordion", flush=True, start_collapsed=True, always_open=True
        ),
    ],
    fluid=True,
    className="pt-4",
)


@callback(
    Output("otherpage-option-accordion", "children"),
    Input("global-ticker-store", "data"),
)
def build_accordion(global_ticker):
    # if no ticker set yet, fall back to AAPL
    ticker = (global_ticker or "AAPL").upper()

    try:
        exps = list_expirations(ticker)
    except Exception as e:
        return [dbc.AccordionItem(str(e), title="Error")]

    today = datetime.today().date()
    future = [e for e in exps if (d := format_date(e)) and d.date() >= today]
    future.sort()

    items = []
    for exp in future:
        label = format_date(exp).strftime("%Y-%m-%d")
        try:
            df = build_option_chain_table(fetch_option_chain(ticker, exp))
        except Exception as e:
            items.append(dbc.AccordionItem(str(e), title=label))
            continue

        # nested header definition
        columns = [
            {"name": ["Calls", "Rho"], "id": "Rho Call"},
            {"name": ["Calls", "Vega"], "id": "Vega Call"},
            {"name": ["Calls", "Theta"], "id": "Theta Call"},
            {"name": ["Calls", "Delta"], "id": "Delta Call"},
            {"name": ["Calls", "IV"], "id": "IV Call"},
            {"name": ["Calls", "Bid"], "id": "Call Bid"},
            {"name": ["Calls", "Ask"], "id": "Call Ask"},
            {"name": ["Strike", ""], "id": "Strike"},
            {"name": ["Puts", "Bid"], "id": "Put Bid"},
            {"name": ["Puts", "Ask"], "id": "Put Ask"},
            {"name": ["Puts", "IV"], "id": "IV Put"},
            {"name": ["Puts", "Delta"], "id": "Delta Put"},
            {"name": ["Puts", "Theta"], "id": "Theta Put"},
            {"name": ["Puts", "Vega"], "id": "Vega Put"},
            {"name": ["Puts", "Rho"], "id": "Rho Put"},
        ]

        table = dash_table.DataTable(
            data=df.to_dict("records"),
            columns=columns,
            merge_duplicate_headers=True,
            fixed_rows={"headers": True},
            style_table={
                "maxHeight": "400px",
                "overflowY": "auto",
                "overflowX": "auto",
                "border": "1px solid #ddd",
            },
            style_header={
                "backgroundColor": "#f8f9fa",
                "fontWeight": "bold",
                "border": "1px solid #ddd",
            },
            style_cell={"textAlign": "center", "padding": "8px", "whiteSpace": "normal"},
            style_data_conditional=[{"if": {"column_id": "Strike"}, "backgroundColor": "#f8d7da"}],
            sort_action="native",
            filter_action="native",
            page_size=9999,
            virtualization=True,
        )

        heatmap = dcc.Graph(figure=build_heatmap(df))

        tabs = dbc.Tabs(
            [
                dbc.Tab(dcc.Loading(table, type="default"), label="Table View"),
                dbc.Tab(dcc.Loading(heatmap, type="default"), label="Heatmap"),
            ]
        )

        items.append(dbc.AccordionItem(tabs, title=label))

    return items
