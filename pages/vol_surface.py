# vol_surface.py
"""
Dash page for visualizing a 3D implied volatility surface using ThetaData.
 - X axis: strike price or moneyness
 - Y axis: time to expiration (years)
 - Z axis: implied volatility (%)
Relies on the global ticker from dcc.Store.
"""

import dash
import dash_bootstrap_components as dbc
from dash import dcc, html, callback, Input, Output, State
import requests
import pandas as pd
import numpy as np
from io import StringIO
from datetime import datetime, date
from scipy.stats import norm
import scipy.optimize as opt
from scipy.interpolate import griddata
from scipy.spatial.qhull import QhullError
import plotly.graph_objects as go

# Register this script as a Dash page
dash.register_page(__name__, path="/vol_surface", title="Vol Surface", name="Vol Surface")


# ------------------------------------------------
# Black‑Scholes & IV Helpers
# ------------------------------------------------

def bs_call_price(S, X, r, T, v, q):
    d1 = (np.log(S/X) + (r - q + 0.5*v*v)*T) / (v*np.sqrt(T))
    d2 = d1 - v*np.sqrt(T)
    return S*np.exp(-q*T)*norm.cdf(d1) - X*np.exp(-r*T)*norm.cdf(d2)

def call_iv(S, X, r, T, price, q):
    if T <= 0 or price <= 0:
        return np.nan
    f = lambda vol: bs_call_price(S, X, r, T, vol, q) - price
    try:
        return opt.brentq(f, 1e-6, 5.0)
    except ValueError:
        return np.nan

def format_date(yyyymmdd):
    try:
        return datetime.strptime(str(yyyymmdd), "%Y%m%d")
    except:
        return None


# ------------------------------------------------
# ThetaData API Helpers
# ------------------------------------------------

def list_expirations(root):
    resp = requests.get(
        "http://127.0.0.1:25510/v2/list/expirations",
        params={"root": root}, timeout=10
    )
    resp.raise_for_status()
    data = resp.json()
    if data.get("header",{}).get("error_type"):
        raise Exception(data["header"]["error_msg"])
    exps = data.get("response",[])
    if not exps:
        raise Exception(f"No expirations for {root}")
    return exps

def fetch_option_chain(root, expiration):
    resp = requests.get(
        "http://127.0.0.1:25510/v2/bulk_snapshot/option/greeks",
        params={"root": root, "exp": expiration, "use_csv": "true"},
        timeout=10
    )
    resp.raise_for_status()
    df = pd.read_csv(StringIO(resp.text))
    if df.empty:
        raise Exception(f"No option data for {root}@{expiration}")
    return df

def detect_scale(df):
    for f in (1,10,100,1000,10000):
        if (df["strike"]/f).between(1,10000).all():
            return f
    return 1

def build_chain(df_raw):
    scale = detect_scale(df_raw)
    df_raw["strike"] = df_raw["strike"].astype(float)/scale
    df_raw["Strike"] = (df_raw["strike"]/10).round(0).astype(int)
    calls = df_raw[df_raw["right"]=="C"]
    calls = calls[["Strike","bid","ask"]].rename(columns={"bid":"Bid","ask":"Ask"})
    return calls.sort_values("Strike")

def fetch_spot(root):
    resp = requests.get(
        "http://127.0.0.1:25510/v2/snapshot/stock/quote",
        params={"root": root, "use_csv": "true"}, timeout=5
    )
    resp.raise_for_status()
    df = pd.read_csv(StringIO(resp.text), header=None)
    bid = float(df.iloc[-1, 3])
    ask = float(df.iloc[-1, 7])
    return 0.5 * (bid + ask)


# ------------------------------------------------
# Page Layout
# ------------------------------------------------
layout = dbc.Container(
    [
        dbc.Row([
            dbc.Col([
                dbc.Label("Select Expirations"),
                dcc.Dropdown(id="surface-expirations", multi=True)
            ], md=6),
            dbc.Col([
                dbc.Label("Risk-Free Rate"),
                dcc.Input(id="surface-rf", type="number", min=0, max=1, step=0.0001, value=0.01)
            ], md=3),
            dbc.Col([
                dbc.Label("Dividend Yield"),
                dcc.Input(id="surface-q", type="number", min=0, max=1, step=0.0001, value=0.0)
            ], md=3),
        ], className="mb-3"),
        dbc.Row([
            dbc.Col([
                dbc.Label("X-Axis:"),
                dcc.RadioItems(
                    id="surface-axis", inline=True,
                    options=[
                        {"label":"Strike Price","value":"strike"},
                        {"label":"Moneyness","value":"moneyness"}
                    ],
                    value="strike"
                )
            ], md=4),
            dbc.Col([
                dbc.Label("Strike Range % of Spot"),
                dcc.RangeSlider(
                    id="surface-strike-range",
                    min=50, max=200, step=1, value=[80,120],
                    marks={50:"50%",100:"100%",150:"150%",200:"200%"}
                )
            ], md=8)
        ], className="mb-4"),
        dcc.Loading(dcc.Graph(id="vol-surface-graph"), type="default")
    ],
    fluid=True, className="pt-4"
)


# ------------------------------------------------
# Callbacks
# ------------------------------------------------
@callback(
    Output("surface-expirations","options"),
    Input("global-ticker-store","data")
)
def load_expirations(ticker):
    if not ticker:
        return []
    root = ticker.upper()
    try:
        exps = list_expirations(root)
    except Exception:
        return []
    today = date.today()
    return [
        {"label": format_date(e).strftime("%Y-%m-%d"), "value": e}
        for e in exps
        if (d:=format_date(e)) and d.date()>=today
    ]

@callback(
    Output("vol-surface-graph","figure"),
    [
        Input("global-ticker-store","data"),
        Input("surface-expirations","value"),
        Input("surface-rf","value"),
        Input("surface-q","value"),
        Input("surface-axis","value"),
        Input("surface-strike-range","value")
    ]
)
def update_surface(ticker, exps, r, q, axis, strike_pct):
    if not ticker or not exps:
        return go.Figure()
    root = ticker.upper()

    # Fetch spot price
    try:
        spot = fetch_spot(root)
    except:
        hist = pd.read_csv(StringIO(requests.get(
            "http://127.0.0.1:25510/v2/hist/stock/eod",
            params={"root":root,"use_csv":"true","start_date":"19000101","end_date":datetime.today().strftime("%Y%m%d")},
            timeout=10
        ).text))
        spot = hist["close"].iloc[-1]

    lo_pct, hi_pct = strike_pct
    records = []
    today = date.today()

    for e in exps:
        d = format_date(e)
        if not d:
            continue
        T = (d.date() - today).days / 365.0
        df_raw = fetch_option_chain(root, e)
        df_chain = build_chain(df_raw)
        lo, hi = spot * lo_pct/100, spot * hi_pct/100
        df_chain = df_chain[(df_chain["Strike"]>=lo) & (df_chain["Strike"]<=hi)]
        for _, row in df_chain.iterrows():
            K = row["Strike"]
            price = 0.5 * (row["Bid"] + row["Ask"])
            iv = call_iv(spot, K, r, T, price, q)
            if np.isnan(iv):
                continue
            records.append({
                "Strike": K,
                "Moneyness": spot / K,
                "T": T,
                "IV": iv * 100
            })

    df = pd.DataFrame(records)
    if df.empty:
        return go.Figure()

    # Prepare grid
    if axis == "moneyness":
        X, Y = df["T"], df["Moneyness"]
    else:
        X, Y = df["T"], df["Strike"]
    Z = df["IV"]

    # If insufficient dimensionality, show 2D line
    if len(np.unique(X)) < 2 or len(np.unique(Y)) < 2:
        fig2d = go.Figure(go.Scatter(x=Y, y=Z, mode="lines+markers", name="IV"))
        fig2d.update_layout(
            title=f"IV Slice: {root}",
            xaxis_title="Moneyness" if axis=="moneyness" else "Strike",
            yaxis_title="Implied Vol (%)"
        )
        return fig2d

    xi = np.linspace(X.min(), X.max(), 30)
    yi = np.linspace(Y.min(), Y.max(), 30)
    xi, yi = np.meshgrid(xi, yi)
    try:
        zi = griddata((X, Y), Z, (xi, yi), method="linear")
    except QhullError:
        zi = griddata((X, Y), Z, (xi, yi), method="nearest")

    fig = go.Figure(go.Surface(x=xi, y=yi, z=zi, colorscale="Viridis", colorbar=dict(title="IV %")))
    fig.update_layout(
        title=f"Implied Vol Surface: {root}",
        scene=dict(
            xaxis_title="Time to Expiry (yrs)",
            yaxis_title="Moneyness" if axis=="moneyness" else "Strike",
            zaxis_title="Implied Vol (%)"
        ),
        margin=dict(l=50, r=50, t=50, b=50)
    )
    return fig
