import numpy as np
import pandas as pd
from src.optimizer import simulate_portfolio
from src.lstm_forecaster import forecast_lstm_returns
from src.gru_forecaster import forecast_gru_returns

def backtest(daily_returns,tickers,lookback=252, freq=63, simulations=10000,
             method='classical'):
    dates = daily_returns.index
    portfolio_returns, portfolio_dates, weights_history = [], [], []
    windows = range(lookback, len(dates) - freq, freq)
    total_windows = len(windows)

    for total, i in enumerate(windows, start=1):
        print(f"Window {total}/{total_windows}")
        train_data = daily_returns.iloc[i - lookback:i]
        test_data = daily_returns.iloc[i:i + freq]

        if method == 'classical':
            expected_returns = train_data.mean() * 252
        elif method == 'lstm':
            expected_returns = forecast_lstm_returns(train_data)
        elif method == 'gru':
            expected_returns = forecast_gru_returns(train_data)
        _, _, summary, weight_df = simulate_portfolio(expected_returns,train_data,simulations)
        weights = weight_df['Best_Weights'].values

        period_returns = test_data.values @ weights
        portfolio_returns.extend(period_returns)
        portfolio_dates.extend(test_data.index)
        weights_history.append({"date": dates[i], "weights": weights})

    result = pd.Series(portfolio_returns, index=portfolio_dates)
    return result, weights_history