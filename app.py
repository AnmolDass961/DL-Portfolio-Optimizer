import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st

from src.data_loader import fetch_prices, get_returns
from src.backtest import backtest
from src.metrics import backtest_metrics

MIN_DATE = pd.to_datetime("2023-01-01")
MAX_TICKERS = 3

@st.cache_data
def cached_fetch_prices(tickers, start_date, end_date):
    return fetch_prices(tickers, start_date, end_date)

@st.cache_data(show_spinner=False)
def cached_backtest(daily_returns, tickers, lookback, freq, method):
    return backtest(daily_returns, tickers, lookback=lookback, freq=freq, method=method)

st.set_page_config(page_title="DL Portfolio Optimizer", layout="wide")
st.title("DL-Driven Portfolio Optimizer")
st.caption("Classical Markowitz - LSTM - GRU forecasted returns, backtested walk-forward.")

st.sidebar.header("Settings")

default_tickers = "AAPL,MSFT,GOOGL"
tickers_input = st.sidebar.text_input(f"Tickers (comma-separated, max {MAX_TICKERS})", default_tickers)
tickers = [t.strip().upper() for t in tickers_input.split(",") if t.strip()]

if len(tickers) == 0:
    st.sidebar.warning("Enter at least 1 ticker.")
    st.stop()
if len(tickers) > MAX_TICKERS:
    st.sidebar.error(f"Max {MAX_TICKERS} tickers allowed — you entered {len(tickers)}. Using the first {MAX_TICKERS}.")
    tickers = tickers[:MAX_TICKERS]

start_date = st.sidebar.date_input("Start date", MIN_DATE, min_value=MIN_DATE)
end_date = st.sidebar.date_input("End date", pd.to_datetime("2024-12-31"), min_value=MIN_DATE)

if pd.to_datetime(start_date) < MIN_DATE:
    st.sidebar.error("Start date can't be before 2023-01-01 — DL methods need bounded runtime.")
    st.stop()
if pd.to_datetime(end_date) <= pd.to_datetime(start_date):
    st.sidebar.error("End date must be after start date.")
    st.stop()

lookback = st.sidebar.slider("Lookback (days)", 63, 252, 126, step=21)
freq = st.sidebar.slider("Rebalance frequency (days)", 21, 126, 63, step=21)

methods = st.sidebar.multiselect(
    "Methods to compare", ["classical", "lstm","gru"])
if not methods:
    st.sidebar.warning("Select at least one method.")
    st.stop()

run = st.sidebar.button("Run Backtest")
if run:
    with st.spinner("Fetching price data..."):
        prices = cached_fetch_prices(tuple(tickers), str(start_date), str(end_date))
        daily_returns = get_returns(prices)

    results = {}
    weights_histories = {}
    for method in methods:
        with st.spinner(f"Running {method} backtest..."):
            returns_series, wh = cached_backtest(
                daily_returns, tuple(tickers), lookback, freq, method
            )
            results[method] = returns_series
            weights_histories[method] = wh

    st.success("Backtest complete.")

    st.subheader("Performance Summary")
    summary_df = pd.DataFrame([
        {"method": m, **backtest_metrics(r)} for m, r in results.items()
    ])
    st.dataframe(summary_df.style.format({
        "annual_return": "{:.2%}", "annual_volatility": "{:.2%}", "sharpe_ratio": "{:.3f}"
    }))
    st.download_button(
        "Download results CSV",
        summary_df.to_csv(index=False),
        "backtest_summary.csv",
        "text/csv"
    )

    st.subheader("Cumulative Portfolio Value")
    fig, ax = plt.subplots(figsize=(10, 5))
    for method, returns_series in results.items():
        np.exp(returns_series.cumsum()).plot(ax=ax, label=method)
    ax.legend()
    ax.set_ylabel("Portfolio Value (normalized)")
    st.pyplot(fig)
    st.subheader("Final Portfolio Weights by Method")

    weights_table = pd.DataFrame(
        {method: weights_histories[method][-1]["weights"] for method in methods},
        index=tickers
    )
    st.dataframe(weights_table.style.format("{:.2%}"))

    st.download_button(
        "Download weights CSV",
        weights_table.to_csv(),
        "final_weights.csv",
        "text/csv"
    )

    fig, ax = plt.subplots(figsize=(10, 4))
    width = 0.8 / len(methods)
    x = np.arange(len(tickers))
    for idx, method in enumerate(methods):
        ax.bar(x + idx * width, weights_table[method], width=width, label=method)
    ax.set_xticks(x + width * (len(methods) - 1) / 2)
    ax.set_xticklabels(tickers)
    ax.legend()
    ax.set_ylabel("Weight")
    st.pyplot(fig)

    st.subheader("Drawdown")
    fig, ax = plt.subplots(figsize=(10, 3))
    for method, returns_series in results.items():
        wealth = np.exp(returns_series.cumsum())
        dd = wealth / wealth.cummax() - 1
        dd.plot(ax=ax, label=method)
    ax.legend()
    ax.set_ylabel("Drawdown")
    st.pyplot(fig)

    st.subheader("Weight Evolution Over Time")
    selected_method = st.selectbox("View weight evolution for:", methods)
    wh = weights_histories[selected_method]
    weight_df = pd.DataFrame(
        [w["weights"] for w in wh],
        index=[w["date"] for w in wh],
        columns=tickers
    )

    fig, ax = plt.subplots(figsize=(10, max(2, 0.4 * len(tickers))))
    im = ax.imshow(weight_df.T, aspect="auto", cmap="YlGnBu", vmin=0, vmax=weight_df.values.max())
    ax.set_yticks(range(len(tickers)))
    ax.set_yticklabels(tickers)
    ax.set_xticks(range(len(weight_df)))
    ax.set_xticklabels([d.strftime("%Y-%m-%d") for d in weight_df.index], rotation=45, ha="right")

    for i in range(len(tickers)):
        for j in range(len(weight_df)):
            ax.text(j, i, f"{weight_df.iloc[j, i]:.0%}", ha="center", va="center",
                    color="black" if weight_df.iloc[j, i] < weight_df.values.max()*0.7 else "white",
                    fontsize=8)

    fig.colorbar(im, ax=ax, label="Weight")
    st.pyplot(fig)

else:
    st.info(f"Set your tickers (max {MAX_TICKERS}) and date range (2023+) in the sidebar, then click **Run Backtest**.")