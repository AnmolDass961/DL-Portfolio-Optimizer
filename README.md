# DL-Driven Portfolio Optimizer

A modernized take on the classical Markowitz mean-variance framework: instead of relying purely on historical average returns, this project uses LSTM,GRU to forecast expected returns, then feeds those forecasts into a Markowitz-style mean-variance optimizer. Performance is evaluated with a  backtest against a classical historical-mean baseline.

## Motivation

Traditional Markowitz optimization estimates expected returns as the simple historical average — a method known to be noisy and highly sensitive to estimation error. This project tests whether a time-series forecasting model (LSTM,GRU) can produce better return estimates, and whether that translates into better risk-adjusted portfolio performance.

## Methodology

1. **Data**: Daily adjusted close prices pulled via `yfinance` for a basket of large-cap tickers, converted to log returns.
2. **Optimization**: A Monte Carlo mean-variance optimizer — thousands of random long-only weight combinations are simulated, and the portfolio with the highest Sharpe ratio is selected. Covariance is estimated classically (historical) for both methods below.
3. **Return forecasting — two approaches compared**:
   - **Classical**: expected return = trailing historical mean (annualized).
   - **LSTM**: a single-layer LSTM per ticker, trained on rolling windows of past daily returns, predicts the next-period return (annualized).
   - **GRU**: a single-layer GRU per ticker, trained on rolling windows of past daily returns, predicts the next-period return (annualized).
4. **Backtesting**: Walk-forward, out-of-sample evaluation — at each rebalance date, only data *prior* to that date is used to compute weights, which are then held and evaluated on the following period. This avoids lookahead bias and simulates how the strategy would have run live.

## Repo Structure

```
dl-portfolio-optimizer/
├── README.md
├── requirements.txt
├── app.py                                         
├── notebooks/
│   └── explore.ipynb             
├── results/
│   └── backtest_summary.csv
├── src/
│   ├── data_loader.py             
│   ├── optimizer.py                
│   ├── lstm_forecaster.py         
│   ├── backtest.py               
│   └── metrics.py                 
```

## Results

Walk-forward backtest, classical vs. LSTM-forecasted returns:

| Method    | Annual Return | Annual Volatility | Sharpe Ratio |Max DrawDown
|-----------|---------------|--------------------|--------------|--------------|
| Classical (Historical Mean) | 12.69% | 22.69% | 0.3389 |-0.145 |
| LSTM Forecast | 22.24% | 23.67% | 0.7286 | -0.2031 |
| GRU Forecast | |11.33% | 16.94% | 0.374 | -0.1483 |


The LSTM-forecasted approach modestly outperformed the classical baseline and GRU-forecast on a risk-adjusted basis,meaning the improvement came from better-timed return forecasts rather than a different risk profile.


## Key Findings

- **The DL approach provided a real but modest edge** over the naive historical-mean baseline — consistent with the well-documented difficulty of forecasting daily equity returns, which behave close to a random walk.
- **Results are sensitive to the evaluation window.** Backtests over shorter, more recent periods showed more volatile and sometimes reversed results compared to the full multi-year backtest above — a reminder that a single backtest window shouldn't be over-interpreted as proof a strategy "works" or "doesn't work."

## Tech Stack

Python, pandas, NumPy, yfinance, TensorFlow/Keras, Streamlit, Matplotlib