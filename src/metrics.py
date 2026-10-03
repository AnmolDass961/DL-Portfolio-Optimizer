import numpy as np

def backtest_metrics(returns, risk_free_rate=0.05):
    ann_return = returns.mean() * 252
    ann_vol = returns.std() * np.sqrt(252)
    sharpe = (ann_return - risk_free_rate) / ann_vol
    wealth = np.exp(returns.cumsum())
    drawdown = wealth / wealth.cummax() - 1
    max_drawdown = drawdown.min()
    return {
        "annual_return": ann_return,
        "annual_volatility": ann_vol,
        "sharpe_ratio": sharpe,
        "max_drawdown": max_drawdown}
