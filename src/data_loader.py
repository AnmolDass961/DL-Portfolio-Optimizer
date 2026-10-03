import pandas as pd
import numpy as np
import yfinance as yf

def fetch_prices(tickers,start_date,end_date):
    data=yf.download(tickers=tickers,start=start_date,end=end_date,progress=False)
    prices=data['Close'].copy()
    return prices

def get_returns(prices):
    return np.log(prices/prices.shift(1)).dropna()