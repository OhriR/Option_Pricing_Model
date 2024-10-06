# -*- coding: utf-8 -*-
"""
Created on Wed Oct  2 02:17:18 2024

Author: Riya Ohri
"""

import yfinance as yf
import pandas as pd
import numpy as np
from scipy.stats import norm
import matplotlib.pyplot as plt
import seaborn as sns
import sys  # Import sys for sys.exit()

# Set the plotting style
sns.set_style('whitegrid')

# Define the ticker symbol
ticker_symbol = 'AAPL'

# Create a Ticker object
ticker = yf.Ticker(ticker_symbol)

# Get available expiration dates
expirations = ticker.options
print("Available expiration dates:", expirations)

# Choose the first expiration date that is in the future
selected_expiration = None
T = None

current_datetime = pd.to_datetime('today').normalize()
print(f"\nCurrent datetime: {current_datetime}")

for date in expirations:
    expiration_datetime = pd.to_datetime(date).normalize()
    delta = (expiration_datetime - current_datetime).days
    print(f"Checking expiration date: {date}, Delta (days): {delta}")
    if delta > 0:
        selected_expiration = date
        T = delta / 365
        print(f"Selected expiration date: {selected_expiration}, Time to expiration (T): {T:.4f} years\n")
        break

if selected_expiration is None:
    print("No future expiration dates available.")
    sys.exit()

# Get the option chain for the selected expiration date
option_chain = ticker.option_chain(selected_expiration)

# Extract calls and puts
calls = option_chain.calls
puts = option_chain.puts

# Display the first few rows
print("Sample Call Options:")
print(calls[['contractSymbol', 'strike', 'lastPrice', 'impliedVolatility']].head())

# Get current stock price
current_stock_price = ticker.history(period='1d')['Close'].iloc[-1]
print(f"\nCurrent stock price for {ticker_symbol}: ${current_stock_price:.2f}")

# Choose an option contract (e.g., the one with the strike price closest to the current stock price)
selected_option = calls.iloc[(calls['strike'] - current_stock_price).abs().argsort()[:1]]
implied_volatility = selected_option['impliedVolatility'].values[0]
print(f"Implied volatility from option chain: {implied_volatility:.2%}")

# Get historical stock prices for the past year
historical_prices = ticker.history(period='1y')
# Calculate daily returns
historical_prices['Returns'] = historical_prices['Close'].pct_change()
# Calculate annualized volatility
historical_volatility = historical_prices['Returns'].std() * np.sqrt(252)
print(f"Historical volatility: {historical_volatility:.2%}")

# Stock price (S)
S = current_stock_price

# Strike price (K)
K = selected_option['strike'].values[0]

# Time to expiration (T) in years
print(f"\nSelected expiration date: {selected_expiration}")
print(f"Time to expiration (T): {T:.4f} years")

# Risk-free interest rate (r)
r = 0.05  # Assume 5% annual risk-free rate
print(f"Risk-free interest rate: {r:.2%}")

# Define a strike price range around the current stock price
strike_range = 20  # +/- $20

# Filter options within the range
filtered_calls = calls[(calls['strike'] >= S - strike_range) & (calls['strike'] <= S + strike_range)]
filtered_calls = filtered_calls.sort_values('strike').reset_index(drop=True)

# Ensure there are options in the filtered list
if filtered_calls.empty:
    print("No call options found within the specified strike range.")
    sys.exit()

print(f"\nNumber of filtered call options: {len(filtered_calls)}")
print("Filtered Call Options:")
print(filtered_calls[['strike', 'lastPrice', 'impliedVolatility']].head())

# Volatility (sigma)
# Use implied volatility or historical volatility if implied volatility is missing
sigma = implied_volatility if not pd.isna(implied_volatility) else historical_volatility

# Define the Black-Scholes model
def black_scholes_call(S, K, T, r, sigma):
    """
    Calculate the Black-Scholes price for a European call option.
    """
    if T <= 0:
        return max(S - K, 0)  # Intrinsic value
    d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    call_price = S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
    return call_price

# Define the Binomial Tree model
def binomial_tree_call(S, K, T, r, sigma, N=100):
    """
    Calculate the Binomial Tree price for a European call option.
    """
    if T <= 0:
        return max(S - K, 0)  # Intrinsic value
    dt = T / N
    if dt <= 0:
        return max(S - K, 0)
    u = np.exp(sigma * np.sqrt(dt))
    d = 1 / u
    p = (np.exp(r * dt) - d) / (u - d)
    discount = np.exp(-r * dt)
    
    # Initialize asset prices at maturity
    ST = np.array([S * (u ** j) * (d ** (N - j)) for j in range(N + 1)])
    # Initialize option values at maturity
    option_values = np.maximum(ST - K, 0)
    
    # Step back through the tree
    for i in range(N - 1, -1, -1):
        option_values = discount * (p * option_values[1:] + (1 - p) * option_values[:-1])
    
    return option_values[0]

# Define the Monte Carlo simulation
def monte_carlo_call(S, K, T, r, sigma, num_simulations=10000):
    """
    Calculate the Monte Carlo price for a European call option.
    """
    if T <= 0:
        return max(S - K, 0)  # Intrinsic value
    np.random.seed(42)  # For reproducibility
    Z = np.random.standard_normal(num_simulations)
    ST = S * np.exp((r - 0.5 * sigma ** 2) * T + sigma * np.sqrt(T) * Z)
    payoffs = np.maximum(ST - K, 0)
    call_price = np.exp(-r * T) * np.mean(payoffs)
    return call_price

# Initialize lists
strikes = []
market_prices = []
bs_prices = []
bt_prices = []
mc_prices = []
implied_vols = []

# Loop through filtered options
print("\nProcessing options and calculating model prices:")
for index, option in filtered_calls.iterrows():
    K = option['strike']
    market_price = option['lastPrice']
    implied_vol = option['impliedVolatility']
    
    # Skip options with missing data
    if pd.isna(market_price) or pd.isna(implied_vol):
        print(f"Skipping strike {K}: Missing market price or implied volatility.")
        continue
    
    # Skip options with zero market price
    if market_price == 0:
        print(f"Skipping strike {K}: Market price is zero.")
        continue
    
    # Use implied volatility or historical volatility if implied volatility is missing
    sigma = implied_vol if not pd.isna(implied_vol) else historical_volatility
    
    # Calculate model prices
    bs_price = black_scholes_call(S, K, T, r, sigma)
    bt_price = binomial_tree_call(S, K, T, r, sigma, N=100)
    mc_price = monte_carlo_call(S, K, T, r, sigma, num_simulations=10000)
    
    # Check for invalid model prices
    if any(map(lambda x: np.isnan(x) or np.isinf(x), [bs_price, bt_price, mc_price])):
        print(f"Invalid model price detected for strike {K}. Skipping this option.")
        continue  # Skip this option if any model price is invalid
    
    # Store strike and prices
    strikes.append(K)
    market_prices.append(market_price)
    bs_prices.append(bs_price)
    bt_prices.append(bt_price)
    mc_prices.append(mc_price)
    implied_vols.append(implied_vol * 100)  # Convert to percentage
    
    # Debugging: Print the appended strike and prices
    print(f"Appended strike: {K}, Market Price: {market_price}, BS Price: {bs_price:.2f}, BT Price: {bt_price:.2f}, MC Price: {mc_price:.2f}")

# Ensure that lists are not empty
if not strikes:
    print("No valid options to process after filtering.")
    sys.exit()

# Calculate Errors
bs_errors = [(bs - mp) / mp * 100 for bs, mp in zip(bs_prices, market_prices)]
bt_errors = [(bt - mp) / mp * 100 for bt, mp in zip(bt_prices, market_prices)]
mc_errors = [(mc - mp) / mp * 100 for mc, mp in zip(mc_prices, market_prices)]

# Debugging: Print lengths of lists
print(f"\nLength of strikes: {len(strikes)}")
print(f"Length of bs_errors: {len(bs_errors)}")
print(f"Length of bt_errors: {len(bt_errors)}")
print(f"Length of mc_errors: {len(mc_errors)}")

# Ensure all error lists have the same length as strikes
assert len(bs_errors) == len(strikes), "Black-Scholes errors length mismatch."
assert len(bt_errors) == len(strikes), "Binomial Tree errors length mismatch."
assert len(mc_errors) == len(strikes), "Monte Carlo errors length mismatch."

# Plot Option Prices vs. Strike Prices
plt.figure(figsize=(12, 6))
plt.plot(strikes, market_prices, label='Market Price', marker='o')
plt.plot(strikes, bs_prices, label='Black-Scholes', marker='x')
plt.plot(strikes, bt_prices, label='Binomial Tree', marker='^')
plt.plot(strikes, mc_prices, label='Monte Carlo', marker='s')
plt.title('Option Prices vs. Strike Prices')
plt.xlabel('Strike Price ($)')
plt.ylabel('Option Price ($)')
plt.legend()
plt.grid(True)
plt.show()

# Plot Errors
plt.figure(figsize=(12, 6))
plt.plot(strikes, bs_errors, label='Black-Scholes Error', marker='x')
plt.plot(strikes, bt_errors, label='Binomial Tree Error', marker='^')
plt.plot(strikes, mc_errors, label='Monte Carlo Error', marker='s')
plt.title('Percentage Error vs. Strike Prices')
plt.xlabel('Strike Price ($)')
plt.ylabel('Percentage Error (%)')
plt.legend()
plt.grid(True)
plt.axhline(0, color='black', linewidth=0.5)
plt.show()

# Plot the Volatility Smile
plt.figure(figsize=(12, 6))
plt.plot(strikes, implied_vols, marker='o', linestyle='-')
plt.title('Implied Volatility Smile')
plt.xlabel('Strike Price ($)')
plt.ylabel('Implied Volatility (%)')
plt.grid(True)
plt.show()

# Define a range of stock prices at expiration
S_range = np.linspace(S - strike_range, S + strike_range, 100)
K_mid = filtered_calls['strike'].iloc[len(filtered_calls) // 2]  # Use a middle strike price

# Calculate call option payoff
call_payoff = np.maximum(S_range - K_mid, 0)

plt.figure(figsize=(12, 6))
plt.plot(S_range, call_payoff, label=f'Call Option Payoff (K=${K_mid})')
plt.title('Call Option Payoff at Expiration')
plt.xlabel('Stock Price at Expiration ($)')
plt.ylabel('Payoff ($)')
plt.legend()
plt.grid(True)
plt.show()
