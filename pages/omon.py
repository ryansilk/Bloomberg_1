# pages/omon.py

import dash
import dash_bootstrap_components as dbc
from dash import html, dash_table, dcc, Input, Output, callback
import requests
import pandas as pd
import numpy as np
from io import StringIO
from datetime import datetime, date, timedelta
import math

dash.register_page(
    __name__,
    path="/omon",
    title="Options Monitor",
    name="OMON"
)

BASE_URL = "http://127.0.0.1:25510/v2"

# ------------------------------
# Helper Functions
# ------------------------------

def list_expirations(root: str):
    resp = requests.get(f"{BASE_URL}/list/expirations", params={"root": root}, timeout=10)
    resp.raise_for_status()
    js = resp.json()
    if js["header"].get("error_type"):
        raise Exception(js["header"]["error_msg"])
    return js["response"] or []

def fetch_snapshot_quote(root: str):
    resp = requests.get(
        f"{BASE_URL}/snapshot/stock/quote",
        params={"root": root, "use_csv": "true"},
        timeout=5
    )
    resp.raise_for_status()
    df = pd.read_csv(StringIO(resp.text), header=None)
    last = df.iloc[-1]
    bid, ask = float(last[3]), float(last[7])
    return bid, ask, 0.5 * (bid + ask)

def fetch_option_chain(root: str, exp: str):
    resp = requests.get(
        f"{BASE_URL}/bulk_snapshot/option/greeks",
        params={"root": root, "exp": exp, "use_csv": "true"},
        timeout=10
    )
    resp.raise_for_status()
    df = pd.read_csv(StringIO(resp.text))
    if df.empty:
        raise Exception("No option data")
    return df

def fetch_eod_closes(root: str, days_back: int = 60) -> pd.Series:
    """Fetches EOD closes for the past days_back*2 calendar days."""
    end = date.today() - timedelta(days=1)
    start = end - timedelta(days=days_back*2)
    resp = requests.get(
        f"{BASE_URL}/hist/stock/eod",
        params={
            "root": root,
            "start_date": start.strftime("%Y%m%d"),
            "end_date": end.strftime("%Y%m%d"),
            "use_csv": "false"
        },
        timeout=10
    )
    resp.raise_for_status()
    js = resp.json()
    rows = js.get("response") or []
    cols = js["header"]["format"]
    df = pd.DataFrame(rows, columns=cols)
    df.rename(columns={"date": "Date", "close": "Close"}, inplace=True)
    df["Date"] = pd.to_datetime(df["Date"], format="%Y%m%d")
    df.sort_values("Date", inplace=True)
    return df["Close"]

def build_chain_table(df_raw: pd.DataFrame) -> pd.DataFrame:
    # Always scale strike by 10
    df_raw["strike"] = df_raw["strike"].astype(float) / 1000
    df_raw["Strike"] = df_raw["strike"].round(2)

    calls = df_raw[df_raw["right"] == "C"].copy()
    puts  = df_raw[df_raw["right"] == "P"].copy()

    calls.rename(columns={
        "theta": "Theta Call",
        "vega": "Vega Call",
        "delta": "Delta Call",
        "implied_vol": "IV Call",
        "bid": "Call Bid",
        "ask": "Call Ask"
    }, inplace=True)
    puts.rename(columns={
        "bid": "Put Bid",
        "ask": "Put Ask",
        "implied_vol": "IV Put",
        "delta": "Delta Put",
        "vega": "Vega Put",
        "theta": "Theta Put"
    }, inplace=True)

    # Ensure expected columns exist
    for col in ["Theta Call", "Vega Call", "Delta Call", "IV Call", "Call Bid", "Call Ask", "Strike"]:
        if col not in calls:
            calls[col] = np.nan
    for col in ["Strike", "Put Bid", "Put Ask", "IV Put", "Delta Put", "Vega Put", "Theta Put"]:
        if col not in puts:
            puts[col] = np.nan

    calls = calls[["Theta Call", "Vega Call", "Delta Call", "IV Call", "Call Bid", "Call Ask", "Strike"]]
    puts  = puts[["Strike", "Put Bid", "Put Ask", "IV Put", "Delta Put", "Vega Put", "Theta Put"]]

    merged = pd.merge(calls, puts, on="Strike", how="outer").sort_values("Strike")
    return merged

# ------------------------------
# Layout
# ------------------------------

layout = dbc.Container(fluid=True, className="pt-4 px-3", children=[

    dbc.Row([
        dbc.Col([
            html.Label("Strikes Around ATM:"),
            dcc.Input(
                id="omon-strike-count",
                type="number",
                min=1, step=1,
                value=5,
                style={"width": "80px"}
            )
        ], width="auto"),

        dbc.Col(html.Div(id="metric-midprice"), width="auto", className="ms-4"),
        dbc.Col(html.Div(id="metric-change"),   width="auto", className="ms-4"),
        dbc.Col(html.Div(id="metric-hv30"),     width="auto", className="ms-4"),
        dbc.Col(html.Div(id="metric-ivatm"),    width="auto", className="ms-4"),
        dbc.Col(html.Div(id="metric-skew"),     width="auto", className="ms-4"),
    ], className="mb-3 align-items-center"),

    html.Div(
        id="omon-chain-container",
        style={"maxHeight": "80vh", "overflowY": "auto"}
    )
])

# ------------------------------
# Callbacks
# ------------------------------

@callback(
    [
        Output("metric-midprice", "children"),
        Output("metric-change",   "children"),
        Output("metric-hv30",     "children"),
        Output("metric-ivatm",    "children"),
        Output("metric-skew",     "children"),
    ],
    Input("global-ticker-store", "data"),
)
def update_metrics(global_ticker):
    if not global_ticker:
        blank = html.Span("", className="text-muted")
        return blank, blank, blank, blank, blank

    root = global_ticker.upper()

    # 1) Mid price
    try:
        bid, ask, mid = fetch_snapshot_quote(root)
        mid_txt = f"Mid: ${mid:,.2f}"
    except Exception:
        mid_txt = "Mid: N/A"

    # 2) Day's change
    try:
        closes = fetch_eod_closes(root, days_back=30)
        if len(closes) >= 2:
            prev, last = closes.iloc[-2], closes.iloc[-1]
            diff = last - prev
            pct  = diff / prev * 100
            change_txt = f"Δ: {diff:+.2f} ({pct:+.2f}%)"
        else:
            change_txt = "Δ: N/A"
    except Exception:
        change_txt = "Δ: N/A"

    # 4) 30-day HV
    try:
        log_ret = np.log(closes / closes.shift(1)).dropna()
        hv30 = log_ret.rolling(30).std().iloc[-1] * math.sqrt(252) * 100
        hv30_txt = f"HV(30): {hv30:.2f}%"
    except Exception:
        hv30_txt = "HV(30): N/A"

    # 5 & 6) ATM IV & Skew from nearest expiry
    try:
        # pick nearest future expiry
        today = date.today()
        exps = sorted(
            e for e in list_expirations(root)
            if datetime.strptime(str(e), "%Y%m%d").date() >= today
        )
        exp = exps[0]
        df_chain = build_chain_table(fetch_option_chain(root, exp))
        idx = (df_chain["Strike"] - mid).abs().idxmin()
        iv_call = df_chain.loc[idx, "IV Call"] * 100
        iv_put  = df_chain.loc[idx, "IV Put"]  * 100
        ivatm_txt = f"ATM IV: {iv_call:.2f}%"
        skew_txt  = f"Skew: {iv_put - iv_call:+.2f}%"
    except Exception:
        ivatm_txt = "ATM IV: N/A"
        skew_txt  = "Skew: N/A"

    return (
        html.Span(mid_txt, className="fw-semibold"),
        html.Span(change_txt, className="fw-semibold ms-2"),
        html.Span(hv30_txt, className="fw-semibold ms-2"),
        html.Span(ivatm_txt, className="fw-semibold ms-2"),
        html.Span(skew_txt, className="fw-semibold ms-2"),
    )

@callback(
    Output("omon-chain-container", "children"),
    [
        Input("global-ticker-store", "data"),
        Input("omon-strike-count",   "value")
    ],
)
def render_condensed_chain(global_ticker, strike_count):
    if not global_ticker:
        return html.Div("No ticker selected.", className="text-muted")

    root = global_ticker.upper()
    try:
        _, _, mid = fetch_snapshot_quote(root)
    except:
        mid = None

    today = date.today()
    exps = [
        e for e in list_expirations(root)
        if datetime.strptime(str(e), "%Y%m%d").date() >= today
    ]
    exps.sort(key=lambda e: datetime.strptime(str(e), "%Y%m%d").date())
    exps = exps[:3]

    N = strike_count or 5
    content = []

    for exp in exps:
        exp_date = datetime.strptime(str(exp), "%Y%m%d").date()
        days_to_exp = (exp_date - today).days

        header = html.Div(
            f"{exp_date.strftime('%d-%b-%y')} ({days_to_exp}d)",
            style={
                "fontWeight": "bold",
                "backgroundColor": "#f8f9fa",
                "padding": "6px",
                "borderBottom": "1px solid #ddd"
            }
        )

        try:
            df_chain = build_chain_table(fetch_option_chain(root, exp))
        except Exception as e:
            body = html.Div(f"Error loading data: {e}", style={"color":"red","padding":"6px"})
        else:
            if mid is not None:
                idx = (df_chain["Strike"] - mid).abs().idxmin()
                low = max(0, idx - N//2)
                high = min(len(df_chain), low + N)
                df_show = df_chain.iloc[low:high]
            else:
                df_show = df_chain.head(N)

            columns = [
                {"name": ["Calls", "Theta"],   "id": "Theta Call"},
                {"name": ["Calls", "Vega"],    "id": "Vega Call"},
                {"name": ["Calls", "Delta"],   "id": "Delta Call"},
                {"name": ["Calls", "IV"],      "id": "IV Call"},
                {"name": ["Calls", "Bid"],     "id": "Call Bid"},
                {"name": ["Calls", "Ask"],     "id": "Call Ask"},
                {"name": ["Strike", ""],       "id": "Strike"},
                {"name": ["Puts", "Bid"],      "id": "Put Bid"},
                {"name": ["Puts", "Ask"],      "id": "Put Ask"},
                {"name": ["Puts", "IV"],       "id": "IV Put"},
                {"name": ["Puts", "Delta"],    "id": "Delta Put"},
                {"name": ["Puts", "Vega"],     "id": "Vega Put"},
                {"name": ["Puts", "Theta"],    "id": "Theta Put"},
            ]

            table = dash_table.DataTable(
                data=df_show.to_dict("records"),
                columns=columns,
                merge_duplicate_headers=True,
                page_size=999,
                virtualization=True,
                fixed_rows={"headers": True},
                style_table={"overflowX":"auto"},
                style_data_conditional=[
                    {"if": {"column_id": "Strike"}, "backgroundColor": "#f8d7da"}
                ],
                style_header={"backgroundColor": "#f8f9fa", "fontWeight": "bold"},
                style_cell={"textAlign": "center", "padding": "4px"}
            )
            body = html.Div(table, style={"padding": "8px 0"})

        content.extend([header, body])

    if not content:
        return html.Div("No upcoming expirations.", className="text-muted")
    return content
