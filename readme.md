# Stock Price Simulation and Risk Analysis in Python

## Table of Contents

- [About This Project](#about-this-project)
- [Project File Descriptions](#project-file-descriptions)
- [What is Geometric Brownian Motion?](#what-is-geometric-brownian-motion)
- [How Geometric Brownian Motion Works](#how-geometric-brownian-motion-works)
- [What This Project Does](#what-this-project-does)
- [How It Works (Python)](#how-it-works-python)
- [What is a Monte Carlo Simulation?](#what-is-a-monte-carlo-simulation)
- [How it Works](#how-it-works)
- [Visualizing Risk: Histogram + Path Plot](#visualizing-risk-histogram--path-plot)
- [Monte Carlo Portfolio Simulation Using Matrix Multiplication](#monte-carlo-portfolio-simulation-using-matrix-multiplication)
- [Why Matrix Multiplication?](#why-matrix-multiplication)
- [How the Simulation Works](#how-the-simulation-works)

---

## About This Project

Hi, I’m Reece — a student at the University of Chicago pursuing a B.S. in Computer Science and a B.A. in Economics with a specialization in machine learning and business economics. I’m currently completing my Master’s in Financial Mathematics (MFE). This project simulates and analyzes stock price behavior using historical data and mathematical modeling. It starts with individual stock paths modeled via Geometric Brownian Motion (GBM), scales into Monte Carlo simulations with thousands of paths, and ends with a portfolio-level risk analysis that incorporates real historical correlations and volatility. Value at Risk (VaR), downside probability, and realistic asset interaction are all demonstrated using Python. The project reflects industry-level techniques used in quant research, trading, and risk modeling.

## Project File Descriptions

This repository contains a series of Python files, each building upon the previous to simulate and analyze stock price dynamics and portfolio-level risk.

- **`GBM.py`**: Implements Geometric Brownian Motion to simulate the evolution of a single stock price path based on parameters like drift, volatility, and time horizon. Useful for understanding basic stochastic behavior in financial markets.

- **`Monte_GBM.py`**: Expands the GBM logic to run many simulations (e.g., 10,000 paths) and visualizes the distribution of final stock prices. Demonstrates how randomness plays out across repeated trials — a cornerstone of risk modeling.

- **`Portfolio_Monte_GBM.py`**: Simulates a multi-asset portfolio using dynamically estimated drift, volatility, and historical correlation (via Cholesky decomposition). It scales GBM across assets and paths, visualizes portfolio evolution, and computes portfolio-level VaR and loss probabilities based on real market behavior. All inputs — including start prices, returns, and covariances — are derived from one year of recent market data.

Each script builds a deeper understanding of market behavior under uncertainty — from a single asset's path to diversified portfolio simulations.

---

## What is Geometric Brownian Motion?

GBM is a stochastic process widely used to model the behavior of stock prices over time. It assumes:

* Constant drift (average return)
* Constant volatility
* Log-normal distribution of prices

It’s a core assumption behind models like Black-Scholes, and a building block in Monte Carlo simulations.

---

## How Geometric Brownian Motion Works

At its core, GBM assumes that stock prices grow continuously over time, but with some randomness. We use a formula that captures both the average growth (drift) and the randomness (volatility).

Here’s the equation we use to simulate it:

$$
S_{t+1} = S_t \cdot \exp\left((\mu - \frac{1}{2} \sigma^2)\Delta t + \sigma \sqrt{\Delta t} \cdot Z_t\right)
$$

Where:

* $S_t$: The stock price at time $t$
* $\mu$: The average return (drift)
* $\sigma$: The volatility (how much it varies)
* $\Delta t$: A small time step (like 1/252 for daily steps)
* $Z_t$: A random value drawn from a standard normal distribution (mean 0, standard deviation 1). Think of it like a weighted coin flip that can nudge the stock price up or down by a random amount, based on volatility.

This formula basically says: take the current price, and adjust it using both the average return and some randomness. That’s what gives the simulated path its wiggly, realistic shape.

It’s powerful because it captures the idea that while we can expect growth over time, prices also bounce around in unpredictable ways.


## What This Project Does

* Simulates a single or multiple GBM stock price paths
* Plots them over time
* Demonstrates how volatility and drift affect long-term outcomes

This forms the base for more complex models like Monte Carlo simulations for portfolios, option pricing, or risk modeling, which are discussed later on.

---

## How It Works (Python)

Basic parameters:

```python
S0 = 100          # initial stock price
mu = 0.1          # expected anunual return
sigma = 0.1123    # annual volatility
T = 1             # time in years
N = 252           # number of steps
dt = int(T/N)     # number of steps
```

Simulation:

```python
Z = np.random.normal(0, 1, N)
S = np.zeros(N)
S[0] = S0
for t in range(1, N):
    S[t] = S[t-1] * np.exp((mu - 0.5 * sigma**2)*dt + sigma*np.sqrt(dt)*Z[t])
```

---

## What is a Monte Carlo Simulation?

A Monte Carlo simulation models uncertainty by running the same process (like GBM) many times using random inputs. In finance, this helps us:

- Visualize a range of possible outcomes
- Estimate the probability of gains or losses
- Measure risk under extreme market conditions

Earlier, We simulate a single stock path — but the framework supports scaling to thousands of simulations and portfolio-level analysis which you can see below.

---

## How it works

Basic Parameters:

```python
S0 = 100          # initial stock price
mu = 0.1          # expected anunual return
sigma = 0.1123    # annual volatility
T = 1             # time in years
N = 252           # number of steps
dt = int(T/N)     # number of steps
M = 10000         # number of simulation paths
```

Simulation:
```python
# Set up matrix for all paths
price_paths = np.zeros((M, N+1))
price_paths[:, 0] = S0   # initialize all paths at S0

# Generate all random shocks (Z) at once
Z = np.random.normal(0, 1, size=(M, N))

# Simulate paths
for t in range(1, N+1):
    price_paths[:, t] = price_paths[:, t-1] * np.exp(
        (mu - 0.5 * sigma**2) * dt + sigma * np.sqrt(dt) * Z[:, t-1]
    )
```

---

## Visualizing Risk: Histogram + Path Plot

To analyze the risk in a meaningful way, we can:

- Plot multiple simulated price paths
- Create a histogram of the final prices across all simulations

This lets us estimate downside risk with:

- Value at Risk (VaR) – the worst expected loss at a confidence level
- Probability of Loss – the probability of the stock having returns below the value you entered at

## Monte Carlo Portfolio Simulation Using Matrix Multiplication

In this final section, we simulate a multi-asset portfolio using:

* Geometric Brownian Motion (GBM)
* Monte Carlo simulation (10,000 paths)
* Matrix multiplication to model correlated returns
* Risk metrics like Value at Risk (VaR) and probability of loss

This helps us estimate the range of possible portfolio outcomes more realistically than treating each asset as independent.

---

## Why Matrix Multiplication?

Assets like AAPL, TSLA, and NVDA often move together — meaning their returns are correlated. To simulate this properly:

1. Define a correlation matrix between assets
2. Compute the covariance matrix:

```python
cov_matrix = np.outer(sigma, sigma) * correlation_matrix
```

3. Apply Cholesky decomposition to get a lower-triangular matrix `L` such that:

```
cov_matrix ≈ L @ L.T
```

4. Generate standard normal random shocks `Z` (shape: time steps × assets), then create correlated shocks:

```python
correlated_Z = Z @ L.T
```

This gives us realistic joint behavior between the assets — if Tesla crashes, Nvidia may follow, and our model reflects that.

---

## How the Simulation Works

* Tickers: AAPL, TSLA, NVDA
* Weights: `[0.4, 0.3, 0.3]`
* Drift: annual expected returns `[8%, 12%, 10%]`
* Volatility: annualized `[15%, 20%, 18%]`
* Correlation matrix: manually specified
* Simulation horizon: 252 trading days
* Number of simulations: 10,000
* Portfolio value at start: \$100,000

At each timestep for each simulation, the asset prices evolve via GBM, and we compute the weighted portfolio value.

Note: Drift, volatility, and correlation are not hardcoded — they are calculated dynamically from historical log returns over the past year using Yahoo Finance data. This ensures the model reflects current market behavior for the selected tickers.
