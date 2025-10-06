import numpy as np
import pandas as pd
import cvxpy as cp

def optimize_sharpe(expected_returns, cov_matrix):
    tickers = list(expected_returns.index)

    mu = expected_returns.values
    Sigma = cov_matrix.loc[tickers, tickers].values

    # Define weights
    w = cp.Variable(len(tickers))

    # Objective: maximize expected return
    objective = cp.Maximize(mu @ w)

    # Constraints
    constraints = [
        cp.sum(w) == 1,  # weights sum to 1
        w >= 0           # no shorting
    ]

    # Additional constraint: risk (volatility^2) <= 1
    portfolio_volatility = cp.quad_form(w, Sigma)
    constraints.append(portfolio_volatility <= 1)

    # Solve problem
    problem = cp.Problem(objective, constraints)
    problem.solve()

    return pd.Series(w.value, index=tickers)