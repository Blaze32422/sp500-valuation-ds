import pandas as pd
import matplotlib.pyplot as plt

# 1. Ratio calculations
def calculate_valuation_ratios(info):
    try:
        pe_ratio = info.get("trailingPE", None)
        peg_ratio = info.get("pegRatio", None)
        pb_ratio = info.get("priceToBook", None)
        roe = info.get("returnOnEquity", None)
        eps = info.get("trailingEps", None)
        market_cap = info.get("marketCap", None)
        dividend_yield = info.get("dividendYield", None)

        return {
            "P/E": pe_ratio,
            "PEG": peg_ratio,
            "P/B": pb_ratio,
            "ROE": roe * 100 if roe is not None else None,
            "EPS": eps,
            "Market Cap": market_cap,
            "Dividend Yield (%)": dividend_yield * 100 if dividend_yield else None
        }
    except Exception as e:
        print("Error in calculate_valuation_ratios:", e)
        return {}

# 2. Filtering logic
def apply_filters(df):
    try:
        filtered = df[
            (df["P/E"] < 30) &
            (df["PEG"] < 1.5) &
            (df["ROE"] > 15) &
            (df["EPS"] > 0)
        ]
        return filtered.reset_index(drop=True)
    except Exception as e:
        print("Error in apply_filters:", e)
        return pd.DataFrame()

# 3. Visualizations
def plot_price_history(ticker, hist):
    try:
        plt.figure(figsize=(8, 3))
        plt.plot(hist.index, hist['Close'], label=f'{ticker} Close Price')
        plt.title(f'{ticker} Stock Price History')
        plt.xlabel('Date')
        plt.ylabel('Price (USD)')
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.show()
    except Exception as e:
        print(f"Error plotting price history for {ticker}:", e)

def plot_eps_trend(ticker, info):
    try:
        eps = info.get("trailingEps")
        if eps is not None:
            years = [2020, 2021, 2022, 2023, 2024]
            eps_values = [eps * (0.8 + 0.1 * i) for i in range(len(years))]

            plt.figure(figsize=(6, 3))
            plt.plot(years, eps_values, marker='o', label='EPS Trend')
            plt.title(f'{ticker} EPS Trend (Estimated)')
            plt.xlabel('Year')
            plt.ylabel('EPS (USD)')
            plt.legend()
            plt.grid(True)
            plt.tight_layout()
            plt.show()
    except Exception as e:
        print(f"Error plotting EPS trend for {ticker}:", e)

# 4. Simulated stock data
fake_stock_info = {
    "AAPL": {
        "trailingPE": 28.5,
        "pegRatio": 1.4,
        "priceToBook": 12.3,
        "returnOnEquity": 0.28,
        "trailingEps": 6.05,
        "marketCap": 2500000000000,
        "dividendYield": 0.006
    },
    "MSFT": {
        "trailingPE": 35.2,
        "pegRatio": 2.0,
        "priceToBook": 13.2,
        "returnOnEquity": 0.35,
        "trailingEps": 9.21,
        "marketCap": 2800000000000,
        "dividendYield": 0.009
    },
    "GOOGL": {
        "trailingPE": 23.8,
        "pegRatio": 1.1,
        "priceToBook": 6.5,
        "returnOnEquity": 0.24,
        "trailingEps": 5.91,
        "marketCap": 1800000000000,
        "dividendYield": 0.0
    }
}

# Fake historical data for plot
fake_hist = pd.DataFrame({
    "Close": [150, 160, 170, 165, 175],
}, index=pd.date_range(start="2020-01-01", periods=5, freq="Y"))

# 5. Run the tool
all_data = []

for ticker, info in fake_stock_info.items():
    print(f"Processing {ticker}...")
    ratios = calculate_valuation_ratios(info)
    stock_data = {"Ticker": ticker, **ratios}
    all_data.append(stock_data)

    plot_price_history(ticker, fake_hist)
    plot_eps_trend(ticker, info)

valuation_df = pd.DataFrame(all_data)
filtered_df = apply_filters(valuation_df)

print("\nFiltered Stocks (Value Candidates):")
print(filtered_df)
use the data set for my stock model try to make it rank companies buy best value per eps
