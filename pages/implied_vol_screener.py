import dash
import dash_bootstrap_components as dbc
from dash import dcc, html, dash_table, callback
from dash.dependencies import Input, Output, State
import httpx
import pandas as pd
import numpy as np
from io import StringIO
from datetime import datetime, timedelta
from dash.dash_table.Format import Format, Scheme
from functools import lru_cache
from concurrent.futures import ThreadPoolExecutor

# Register this file as a page in the multipage app.
dash.register_page(__name__, path='/implied_vol_screener', title="Implied Volatility Screener")

BASE_URL = "http://127.0.0.1:25510/v2"


###################################
# Data-Fetching Utility Functions #
###################################

@lru_cache(maxsize=32)
def fetch_expirations(root: str):
    try:
        url = f"{BASE_URL}/list/expirations"
        params = {"root": root}
        response = httpx.get(url, params=params, timeout=10)
        response.raise_for_status()
        data = response.json()
        expirations = data.get("response", [])
        return [str(exp) for exp in expirations]
    except Exception:
        return []


def fetch_greeks(root: str, exp: str, start_date: str, end_date: str) -> pd.DataFrame:
    url = f"{BASE_URL}/bulk_hist/option/eod_greeks"
    params = {
        "root": root,
        "exp": exp,
        "start_date": start_date,
        "end_date": end_date,
        "use_csv": "true",
    }
    r = httpx.get(url, params=params, timeout=10)
    r.raise_for_status()
    return pd.read_csv(StringIO(r.text))


@lru_cache(maxsize=256)
def fetch_index_close(index_symbol: str, date_str: str) -> float:
    url = f"{BASE_URL}/hist/index/eod"
    params = {
        "root": index_symbol,
        "start_date": date_str,
        "end_date": date_str,
        "use_csv": "true",
    }
    try:
        r = httpx.get(url, params=params, timeout=10)
        r.raise_for_status()
        lines = r.text.strip().split("\n")
        if len(lines) < 2:
            return None
        headers = lines[0].split(",")
        try:
            close_idx = headers.index("close")
        except ValueError:
            return None
        data_rows = lines[1:]
        for row in data_rows:
            cols = row.split(",")
            if len(cols) <= close_idx:
                continue
            try:
                return float(cols[close_idx])
            except ValueError:
                pass
        return None
    except httpx.HTTPStatusError:
        return None


def get_index_pct_change(index_symbol: str, dte_date: str, exp_date: str) -> float:
    """Fetch close on dte_date and exp_date for index_symbol, then return % change."""

    def previous_weekday(d: datetime) -> datetime:
        while d.weekday() >= 5:  # Skip weekends (5=Saturday,6=Sunday)
            d -= timedelta(days=1)
        return d

    dte_dt = previous_weekday(datetime.strptime(dte_date, "%Y%m%d"))
    dte_close = None
    for _ in range(5):
        c = fetch_index_close(index_symbol, dte_dt.strftime("%Y%m%d"))
        if c is not None:
            dte_close = c
            break
        dte_dt -= timedelta(days=1)

    exp_dt = previous_weekday(datetime.strptime(exp_date, "%Y%m%d"))
    exp_close = None
    for _ in range(5):
        c = fetch_index_close(index_symbol, exp_dt.strftime("%Y%m%d"))
        if c is not None:
            exp_close = c
            break
        exp_dt -= timedelta(days=1)

    if dte_close is None or exp_close is None or dte_close == 0:
        return np.nan
    return (exp_close - dte_close) / dte_close * 100


###################################
# Core Analysis (Shared Logic)    #
###################################
def run_expiration_analysis(root_value: str, selected_exp: str, analysis_date_dt: datetime, prev_count: int):
    """
    For the given ticker (root_value) and selected expiration, return a DataFrame of analysis data.

    If the user selects an expiration that is multiple slots out in the future,
    we add additional iterations to capture enough historical data.
    """
    all_exps = fetch_expirations(root_value)
    if not all_exps:
        return pd.DataFrame()

    try:
        idx = all_exps.index(selected_exp)
    except ValueError:
        return pd.DataFrame()

    future_exps = [e for e in all_exps if datetime.strptime(e, "%Y%m%d") > analysis_date_dt]
    if selected_exp not in future_exps:
        effective_prev_count = prev_count
    else:
        future_exps_sorted = sorted(future_exps)
        selected_future_idx = future_exps_sorted.index(selected_exp)
        offset = selected_future_idx
        effective_prev_count = prev_count + offset

    exp_dt = datetime.strptime(selected_exp, "%Y%m%d")
    days_to_expiration = (exp_dt - analysis_date_dt).days
    if days_to_expiration <= 0:
        return pd.DataFrame()

    historical_moves = []
    loopcount = 0

    for i in range(idx, max(idx - (effective_prev_count + 1), -1), -1):
        current_exp = all_exps[i]
        current_exp_dt = datetime.strptime(current_exp, "%Y%m%d")

        current_end_date_dt = current_exp_dt - timedelta(days=days_to_expiration)
        current_start_date_dt = current_end_date_dt - timedelta(days=1)
        while current_start_date_dt.weekday() >= 5:
            current_start_date_dt += timedelta(days=1)

        current_end_date_str = current_end_date_dt.strftime("%Y%m%d")
        current_start_date_str = current_start_date_dt.strftime("%Y%m%d")

        try:
            df_current = fetch_greeks(root_value, current_exp, current_start_date_str, current_end_date_str)
        except httpx.HTTPStatusError as e:
            if e.response.status_code == 472:
                historical_moves.append({
                    "Expiration": current_exp,
                    "Ref_Date": "",
                    "Ref_Underlying(DTE)": np.nan,
                    "Exp_Underlying": np.nan,
                    "Move_%": np.nan,
                    "ImpliedMove_%": np.nan,
                    "SPX_Change": np.nan
                })
                continue
            else:
                raise

        df_current["strike"] = df_current["strike"] / 1000.0
        df_current["date"] = pd.to_datetime(df_current["date"], format="%Y%m%d")
        df_current["days_to_expiration"] = (
                datetime.strptime(current_exp, "%Y%m%d") - df_current["date"]
        ).dt.days

        current_dte_data = df_current[df_current["days_to_expiration"] == days_to_expiration].copy()
        ref_underlying = current_dte_data["underlying_price"].mean() if not current_dte_data.empty else np.nan

        if loopcount == 0 and current_exp_dt > analysis_date_dt:
            exp_underlying = ref_underlying
            exp_date_str2 = analysis_date_dt.strftime("%Y%m%d")
        else:
            exp_date_dt2 = current_exp_dt
            while exp_date_dt2.weekday() >= 5:
                exp_date_dt2 -= timedelta(days=1)

            found_data = False
            df_expire_day = pd.DataFrame()
            for _ in range(5):
                tmp_str = exp_date_dt2.strftime("%Y%m%d")
                try:
                    df_try = fetch_greeks(root_value, current_exp, tmp_str, tmp_str)
                    if not df_try.empty and not df_try["underlying_price"].isna().all():
                        df_expire_day = df_try
                        found_data = True
                        break
                except httpx.HTTPStatusError:
                    pass
                exp_date_dt2 -= timedelta(days=1)

            if found_data and not df_expire_day.empty:
                exp_underlying = df_expire_day["underlying_price"].mean()
                exp_date_str2 = exp_date_dt2.strftime("%Y%m%d")
            else:
                exp_underlying = np.nan
                exp_date_str2 = current_exp

        if pd.notna(ref_underlying) and ref_underlying != 0:
            normal_hist_move_pct = (exp_underlying - ref_underlying) / ref_underlying * 100
        else:
            normal_hist_move_pct = np.nan

        df_calls = current_dte_data[current_dte_data["right"] == "C"].copy()
        df_puts = current_dte_data[current_dte_data["right"] == "P"].copy()

        def best_bid(df_opts, underlying):
            if df_opts.empty or pd.isna(underlying):
                return 0.0
            df_under = df_opts[df_opts["strike"] <= underlying].copy()
            if df_under.empty:
                return 0.0
            df_under["diff"] = underlying - df_under["strike"]
            row_bid = df_under.loc[df_under["diff"].idxmin()]
            return row_bid["bid"]

        call_bid_val = best_bid(df_calls, ref_underlying)
        put_bid_val = best_bid(df_puts, ref_underlying)
        if pd.notna(ref_underlying) and ref_underlying != 0:
            implied_move_for_exp = (call_bid_val + put_bid_val) / ref_underlying * 100
        else:
            implied_move_for_exp = 0.0

        move_pct = implied_move_for_exp if loopcount == 0 else normal_hist_move_pct
        spx_change = get_index_pct_change("SPX", current_start_date_str, exp_date_str2)

        historical_moves.append({
            "Expiration": current_exp,
            "Ref_Date": current_end_date_str,
            "Ref_Underlying(DTE)": ref_underlying,
            "Exp_Underlying": exp_underlying,
            "Move_%": move_pct,
            "ImpliedMove_%": implied_move_for_exp,
            "SPX_Change": spx_change
        })

        loopcount += 1

    df_hist = pd.DataFrame(historical_moves)
    if df_hist.empty:
        return pd.DataFrame()

    for col in ["Move_%", "ImpliedMove_%", "SPX_Change"]:
        if col in df_hist.columns:
            df_hist[col] = df_hist[col].round(2)

    if len(df_hist) > 0:
        implied_first = df_hist.loc[0, "Move_%"]
    else:
        implied_first = np.nan

    df_hist["Implied_%_vs_Historical"] = np.where(
        df_hist["Move_%"] == 0,
        np.nan,
        (implied_first / df_hist["Move_%"]) * 100
    ).round(2)
    df_hist["Implied Delta"] = (df_hist["ImpliedMove_%"] - df_hist["Move_%"].abs()).round(2)

    # Rename headers as per the mapping and remove the Ticker column.
    # Mapping:
    # "Expiration" -> "Previous Expiration"
    # "Ref_Date" -> "Expiration Cycle Start"
    # "Ref_Underlying(DTE)" -> "Cycle Start Stock"
    # "Exp_Underlying" -> "Cycle End Stock"
    # "Move_%" -> "Stock Move (%)"
    # "ImpliedMove_%" -> "Implied Move (%)"
    # "SPX_Change" -> "SPX Move (%)"
    # "Implied_%_vs_Historical" -> "IV vs HV"
    # "Implied Delta" -> "Implied Delta (%)"
    df_final = df_hist.rename(columns={
        "Expiration": "Previous Expiration",
        "Ref_Date": "Expiration Cycle Start",
        "Ref_Underlying(DTE)": "Cycle Start Stock",
        "Exp_Underlying": "Cycle End Stock",
        "Move_%": "Stock Move (%)",
        "ImpliedMove_%": "Implied Move (%)",
        "SPX_Change": "SPX Move (%)",
        "Implied_%_vs_Historical": "IV vs HV",
        "Implied Delta": "Implied Delta (%)"
    })

    # Drop rows where "Expiry Underlying" (now "Cycle End Stock") is missing.
    df_final = df_final.dropna(subset=["Cycle End Stock"])

    # Remove the top row.
    if not df_final.empty:
        df_final = df_final.iloc[1:]

    return df_final


###################################
# Ticker Analysis in Parallel     #
###################################
def analyze_ticker(ticker: str, analysis_dt: datetime, prev_count: int, selected_expirations: list):
    all_exps = fetch_expirations(ticker)
    if not all_exps:
        tab_content = html.Div(f"No expiration data available for ticker {ticker}.")
        return dcc.Tab(label=ticker, children=tab_content), []

    future_exps = [exp_str for exp_str in all_exps if datetime.strptime(exp_str, "%Y%m%d") > analysis_dt]
    future_exps.sort()

    if selected_expirations:
        future_exps = [exp for exp in future_exps if exp in selected_expirations]

    if not future_exps:
        tab_content = html.Div(
            f"No selected future expirations are available for ticker {ticker} based on the analysis date."
        )
        return dcc.Tab(label=ticker, children=tab_content), []

    accordion_items = []
    ticker_results = []

    for exp_str in future_exps:
        df_result = run_expiration_analysis(ticker, exp_str, analysis_dt, prev_count)
        if df_result.empty:
            body_content = html.Div(f"No data available for expiration {exp_str}.")
            accordion_items.append(
                dbc.AccordionItem(
                    children=body_content,
                    title=f"Expiration {exp_str}",
                    item_id=f"{ticker}-exp-{exp_str}",
                )
            )
        else:
            # Note: We no longer add a Ticker column.
            # Build table columns based on the final header names.
            columns = []
            numeric_cols = [
                "Stock Move (%)", "Implied Move (%)", "SPX Move (%)", "IV vs HV", "Implied Delta (%)",
                "Cycle Start Stock", "Cycle End Stock"
            ]
            for col in df_result.columns:
                if col in numeric_cols:
                    columns.append({
                        "name": col,
                        "id": col,
                        "type": "numeric",
                        "format": Format(precision=2, scheme=Scheme.fixed)
                    })
                else:
                    columns.append({
                        "name": col,
                        "id": col,
                        "type": "text"
                    })

            # Update conditional styling to use the new numeric column names.
            style_data_conditional = []
            highlight_cols = ["Stock Move (%)", "Implied Delta (%)", "IV vs HV", "SPX Move (%)"]
            for c in highlight_cols:
                style_data_conditional.extend([
                    {
                        "if": {"filter_query": f"{{{c}}} > 0", "column_id": c},
                        "backgroundColor": "#e6f4ea",
                        "color": "green"
                    },
                    {
                        "if": {"filter_query": f"{{{c}}} < 0", "column_id": c},
                        "backgroundColor": "#fde7e7",
                        "color": "red"
                    },
                ])

            table = dash_table.DataTable(
                data=df_result.to_dict("records"),
                columns=columns,
                style_cell={"textAlign": "center", "padding": "5px"},
                style_header={"fontWeight": "bold", "backgroundColor": "#f8f9fa"},
                style_data_conditional=style_data_conditional,
                page_size=10,
            )

            avg_stock_move_pct = df_result["Stock Move (%)"].abs().mean()
            actual_implied_pct = df_result["Implied Move (%)"].iloc[0]
            exp_dt = datetime.strptime(exp_str, "%Y%m%d")
            days_for_item = (exp_dt - analysis_dt).days

            item_header = dbc.Row(
                [
                    dbc.Col(
                        html.Span(
                            f"{days_for_item} days to Expiration",
                            style={"fontSize": "1.1rem"}
                        ),
                        md=4, className="text-left"
                    ),
                    dbc.Col(
                        html.Span(
                            f"Expiration {exp_str}",
                            style={"fontWeight": "bold", "fontSize": "1.2rem"}
                        ),
                        md=4, className="text-center"
                    ),
                    dbc.Col(
                        html.Span(
                            f"Avg Stock Move: {avg_stock_move_pct:.2f}% | Implied: {actual_implied_pct:.2f}%",
                            style={"fontSize": "1.1rem"}
                        ),
                        md=4, className="text-right"
                    ),
                ],
                align="center",
                style={"width": "100%"},
            )

            accordion_items.append(
                dbc.AccordionItem(
                    children=table,
                    title=item_header,
                    item_id=f"{ticker}-exp-{exp_str}"
                )
            )
            ticker_results.append(df_result)

    accordion_component = dbc.Accordion(
        accordion_items,
        start_collapsed=True,
        always_open=True,
        flush=True,
        id=f"accordion-{ticker}"
    )
    return dcc.Tab(label=ticker, children=accordion_component), ticker_results


#######################
# Page Layout         #
#######################
layout = dbc.Container(
    [
        dbc.Card(
            dbc.CardBody(
                [
                    html.H2("Implied Premium Screener", className="text-center mb-3"),
                    html.P(
                        "Enter one or more stock tickers, pick an analysis date, select how many previous expirations to analyze, and optionally select future expirations to inspect.",
                        className="text-center"
                    ),
                    dbc.Row(
                        [
                            dbc.Col(
                                [
                                    dbc.Label("Tickers"),
                                    dbc.Input(
                                        id="ticker-input",
                                        type="text",
                                        placeholder="e.g., AAPL, MSFT, GOOGL",
                                        value="AAPL, MSFT",
                                    ),
                                ],
                                md=3,
                            ),
                            dbc.Col(
                                [
                                    dbc.Label("Analysis Date"),
                                    dcc.DatePickerSingle(
                                        id="analysis-date",
                                        display_format="YYYY-MM-DD",
                                        placeholder="Select date",
                                        style={"width": "100%"},
                                    ),
                                ],
                                md=3,
                            ),
                            dbc.Col(
                                [
                                    dbc.Label("Previous Iterations"),
                                    dbc.Input(
                                        id="prev-count-input",
                                        type="number",
                                        min=1,
                                        step=1,
                                        value=10,
                                        style={"width": "100%"},
                                    ),
                                ],
                                md=3,
                            ),
                            dbc.Col(
                                [
                                    dbc.Label("Future Expirations"),
                                    dcc.Dropdown(
                                        id="expiration-dropdown",
                                        multi=True,
                                        placeholder="Select future expirations"
                                    ),
                                ],
                                md=3,
                            ),
                        ],
                        className="mb-3",
                    ),
                    dbc.Row(
                        [
                            dbc.Col(
                                dbc.Button(
                                    "Run Analysis",
                                    id="run-button",
                                    n_clicks=0,
                                    color="primary",
                                    style={"width": "100%"}
                                ),
                                md=6,
                            ),
                            dbc.Col(
                                dbc.Button(
                                    "Download Combined CSV",
                                    id="download-button",
                                    n_clicks=0,
                                    color="secondary",
                                    style={"width": "100%"}
                                ),
                                md=6,
                            ),
                        ],
                        className="mb-3",
                    ),
                    dcc.Store(id="analysis-data-store"),
                    html.Hr(),
                    dcc.Loading(
                        id="loading-indicator",
                        type="default",
                        children=html.Div(id="results-tabs-container")
                    ),
                    dcc.Download(id="download-csv"),
                ]
            ),
            className="mt-4 shadow",
            style={"borderRadius": "10px"},
        )
    ],
    fluid=True,
    style={"backgroundColor": "#f5f5f5", "minHeight": "100vh", "padding": "20px"},
)


#############################################
# Callback to populate expiration dropdown options
#############################################
@callback(
    Output("expiration-dropdown", "options"),
    [Input("ticker-input", "value"),
     Input("analysis-date", "date")]
)
def update_expiration_options(tickers_input, analysis_date):
    if not tickers_input or not analysis_date:
        return []
    tickers = [ticker.strip().upper() for ticker in tickers_input.split(",") if ticker.strip()]
    if not tickers:
        return []
    ticker = tickers[0]
    all_exps = fetch_expirations(ticker)
    if not all_exps:
        return []
    analysis_dt = datetime.strptime(analysis_date, "%Y-%m-%d")
    future_exps = [exp for exp in all_exps if datetime.strptime(exp, "%Y%m%d") > analysis_dt]
    future_exps.sort()
    options = [{"label": exp, "value": exp} for exp in future_exps]
    return options


#############################################
# Callback to Build the Ticker Tabs and Store Data
#############################################
@callback(
    [Output("results-tabs-container", "children"),
     Output("analysis-data-store", "data")],
    [Input("run-button", "n_clicks")],
    [
        State("ticker-input", "value"),
        State("analysis-date", "date"),
        State("prev-count-input", "value"),
        State("expiration-dropdown", "value")
    ]
)
def update_results(n_clicks, tickers_input, analysis_date, prev_count, selected_expirations):
    if n_clicks < 1 or not tickers_input or not analysis_date:
        return "", None

    if not prev_count or prev_count < 1:
        prev_count = 10

    tickers = [ticker.strip().upper() for ticker in tickers_input.split(",") if ticker.strip()]
    if not tickers:
        return html.Div("No valid tickers entered."), None

    try:
        analysis_dt = datetime.strptime(analysis_date, "%Y-%m-%d")
    except ValueError:
        return html.Div("Invalid analysis date."), None

    if selected_expirations is None:
        selected_expirations = []

    tabs = []
    combined_results = []

    with ThreadPoolExecutor(max_workers=4) as executor:
        futures = [
            executor.submit(analyze_ticker, ticker, analysis_dt, prev_count, selected_expirations)
            for ticker in tickers
        ]
        for future in futures:
            ticker_tab, ticker_results = future.result()
            tabs.append(ticker_tab)
            if ticker_results:
                combined_results.extend(ticker_results)

    tabs_component = dcc.Tabs(children=tabs, value=tickers[0] if tickers else None)

    if combined_results:
        combined_df = pd.concat(combined_results, ignore_index=True)
        store_data = combined_df.to_json(date_format='iso', orient='split')
    else:
        store_data = None

    return tabs_component, store_data


#############################################
# Callback for Downloading Combined CSV
#############################################
@callback(
    Output("download-csv", "data"),
    Input("download-button", "n_clicks"),
    State("analysis-data-store", "data"),
    prevent_initial_call=True,
)
def download_csv(n_clicks, store_data):
    if not store_data:
        return dash.no_update
    df = pd.read_json(store_data, orient='split')
    return dcc.send_data_frame(df.to_csv, "combined_analysis.csv", index=False)
