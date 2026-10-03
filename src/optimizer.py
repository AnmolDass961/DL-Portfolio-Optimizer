import numpy as np
import pandas as pd

def portfolio_performance(weights,returns,cov_mat):
    portfolio_returns=np.dot(weights,returns)
    portfolio_volatility=np.sqrt(weights.T @ cov_mat @ weights)
    return portfolio_returns,portfolio_volatility

def simulate_portfolio(expected_returns,daily_returns,simulations=10000,seed=42):
    risk_free_rate = 0.05
    tickers = list(daily_returns.columns)
    expected_returns = pd.Series(expected_returns).reindex(tickers).values

    weights_list=[]
    cov_mat=daily_returns.cov()*252
    result=np.zeros((3,simulations))

    for i in range(simulations):
        weights=np.random.random(len(tickers))
        weights=weights/np.sum(weights)
        weights_list.append(weights)

        port_return,port_vol=portfolio_performance(weights,expected_returns,cov_mat)
        result[0,i]=port_return
        result[1,i]=port_vol
        result[2,i]=(port_return-risk_free_rate)/port_vol

    idx=np.argmax(result[2])
    best_weights=weights_list[idx]
    summary = {
        "return": result[0, idx],
        "volatility": result[1, idx],
        "sharpe_ratio": result[2, idx]}

    portfolio_weight_df=pd.DataFrame({'Ticker':tickers,'Best_Weights':best_weights})
    return result,weights_list,summary,portfolio_weight_df


