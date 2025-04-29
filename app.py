# app.py

import flask
from flask import session
import dash
import dash_bootstrap_components as dbc
from dash import dcc, html, Input, Output, State, callback_context, exceptions
import requests, csv
from io import StringIO
import pandas as pd
from datetime import datetime

# ------------------------------------------------------------------------------
# Configuration
# ------------------------------------------------------------------------------
BASE_URL = "http://127.0.0.1:25510/v2"

server = flask.Flask(__name__)
server.secret_key = "REPLACE_WITH_A_STRONG_RANDOM_SECRET"

app = dash.Dash(
    __name__,
    server=server,
    use_pages=True,
    external_stylesheets=[dbc.themes.BOOTSTRAP],
)

# ------------------------------------------------------------------------------
# 1) Two “sub-layouts”: login page vs. main app
# ------------------------------------------------------------------------------
def login_page_layout():
    return dbc.Container(
        [
            html.H2("Please Log In", className="my-4 text-center"),
            dbc.Input(id="login-email",    type="email",    placeholder="Email",    className="mb-2"),
            dbc.Input(id="login-password", type="password", placeholder="Password", className="mb-2"),
            dbc.Button("Log In", id="login-button", color="primary", n_clicks=0, className="w-100"),
            html.Div(id="login-alert", className="mt-2"),
        ],
        style={"maxWidth":"400px","marginTop":"100px"}
    )

def main_app_layout():
    top_bar = dbc.Row(
        [
            dbc.Col(dbc.Input(id="top-ticker-input", type="text", placeholder="e.g., AAPL"), width="auto"),
            dbc.Col(dbc.Button("Set Ticker", id="top-ticker-button", color="primary", n_clicks=0), width="auto"),
            dbc.Col(html.Div(id="top-ticker-stats"), width=True),
            dbc.Col(
                dbc.Button("Logout", id="logout-button", color="secondary", n_clicks=0),
                width="auto",
                style={"textAlign":"right"}
            ),
        ],
        align="center",
        className="mb-4",
    )

    return dbc.Container(
        [
            top_bar,
            dcc.Store(id="global-ticker-store", storage_type="memory", data=""),
            html.Div(id="navbar-container"),
            dash.page_container,  # <-- required for dash_pages
        ],
        fluid=True,
        style={"padding":"20px"}
    )

# ------------------------------------------------------------------------------
# 2) Static layout with both wrappers present
# ------------------------------------------------------------------------------
app.layout = html.Div(
    [
        dcc.Location(id="url", refresh=False),
        html.Div(login_page_layout(), id="login-page"),
        html.Div(main_app_layout(),  id="app-page", style={"display":"none"}),
    ]
)

# ------------------------------------------------------------------------------
# 3) Show/hide login vs. app based on session
# ------------------------------------------------------------------------------
@app.callback(
    Output("login-page", "style"),
    Output("app-page",   "style"),
    Input("url", "pathname")
)
def toggle_pages(pathname):
    if session.get("logged_in"):
        return {"display":"none"}, {}
    return {}, {"display":"none"}

# ------------------------------------------------------------------------------
# 4) Combined Login & Logout callback
# ------------------------------------------------------------------------------
@app.callback(
    Output("url",         "pathname"),
    Output("login-alert", "children"),
    Input("login-button", "n_clicks"),
    Input("logout-button","n_clicks"),
    State("login-email",    "value"),
    State("login-password", "value"),
    prevent_initial_call=True
)
def handle_auth(login_n, logout_n, email, pwd):
    ctx = callback_context
    if not ctx.triggered:
        raise exceptions.PreventUpdate

    triggered_id = ctx.triggered[0]["prop_id"].split(".")[0]

    if triggered_id == "login-button":
        if not email or not pwd:
            return dash.no_update, dbc.Alert("Enter both email and password.", color="warning")
        # TODO: add real auth check here
        session["logged_in"]  = True
        session["user_email"] = email
        return "/other_page", None

    elif triggered_id == "logout-button":
        session.clear()
        return "/login", dash.no_update

    raise exceptions.PreventUpdate

# ------------------------------------------------------------------------------
# 5) Data‐fetching helpers & your existing callbacks
# ------------------------------------------------------------------------------
def fetch_current_price(ticker: str) -> float:
    url = f"{BASE_URL}/snapshot/stock/quote"
    params = {"root": ticker.upper(), "use_csv": "true"}
    resp = requests.get(url, params=params, timeout=5)
    resp.raise_for_status()
    reader = csv.reader(StringIO(resp.text.strip()))
    rows = [r for r in reader if r]
    bid, ask = float(rows[-1][3]), float(rows[-1][7])
    return (bid + ask) / 2

def fetch_stock_eod(ticker: str, start_date: str, end_date: str) -> pd.DataFrame:
    url = f"{BASE_URL}/hist/stock/eod"
    params = {
        "root": ticker.upper(),
        "start_date": start_date,
        "end_date": end_date,
        "use_csv": "true"
    }
    resp = requests.get(url, params=params, timeout=10)
    resp.raise_for_status()
    return pd.read_csv(StringIO(resp.text))

def get_stock_price_changes(ticker: str):
    today = datetime.today()
    end   = today.strftime("%Y%m%d")
    start = (today - pd.Timedelta(days=500)).strftime("%Y%m%d")
    df = fetch_stock_eod(ticker, start, end)
    df['date'] = pd.to_datetime(df['date'], format='%Y%m%d')
    df.sort_values('date', inplace=True)
    current = df.iloc[-1]['close']
    windows = {"5-Day":5, "1-Month":22, "6-Month":126, "1-Year":252}
    changes = {}
    for lbl, d in windows.items():
        if len(df) > d:
            past = df.iloc[-(d+1)]['close']
            changes[lbl] = round((current - past) / past * 100, 2)
        else:
            changes[lbl] = None
    return current, changes

@app.callback(
    Output("global-ticker-store",   "data"),
    Output("top-ticker-stats",      "children"),
    Input("top-ticker-button",      "n_clicks"),
    State("top-ticker-input",       "value"),
    prevent_initial_call=True
)
def on_set_ticker(n, ticker_input):
    if not ticker_input:
        return dash.no_update, dbc.Alert("Please enter a ticker.", color="warning")
    t = ticker_input.strip().upper()
    try:
        price, changes = fetch_current_price(t), get_stock_price_changes(t)[1]
        spans = [
            html.Span(f"Current Price: ${price:,.2f}", className="fw-semibold"),
            html.Span(" | ", className="text-muted mx-2")
        ]
        for lbl in ["5-Day","1-Month","6-Month","1-Year"]:
            v = changes[lbl]
            text = f"{lbl} Change: {v}%" if v is not None else f"{lbl} Change: N/A"
            spans.append(html.Span(text, className="fw-semibold"))
            if lbl != "1-Year":
                spans.append(html.Span(" | ", className="text-muted mx-2"))
        card = dbc.Card(
            dbc.CardBody(html.Div(spans, className="d-flex justify-content-center")),
            className="shadow-sm"
        )
        return t, card
    except Exception as e:
        return dash.no_update, dbc.Alert(f"Error fetching data: {e}", color="danger")

@app.callback(
    Output("navbar-container","children"),
    Input("global-ticker-store","data")
)
def update_nav(global_ticker):
    pages = [
        ("Vol Table","/vol_table"), ("Vol Smile","/vol_smile"), ("Vol Surface","/vol_surface"),
        ("Historical Vol","/historical_vol"), ("Term Structure","/term_structure"),
        ("Monitor","/omon"), ("Option Chain","/other_page"),
        ("IV Screen","/implied_vol_screener"), ("Historical Greeks","/option_viewer"),
    ]
    links = []
    for label, path in pages:
        href = f"{path}?ticker={global_ticker}" if global_ticker else path
        links.append(dbc.NavLink(label, href=href, active="exact"))
    return dbc.Nav(links, pills=True, className="mb-3")

# ------------------------------------------------------------------------------
if __name__ == "__main__":
    app.run_server(debug=True, port=8080)
