import os
import time
import datetime
import numpy as np
import pandas as pd
import requests
from scipy.signal import find_peaks
from dotenv import load_dotenv

def fetch_data_for_symbol(symbol, base_url, api_token):
    """
    Fetch historical stock price data for a given symbol from Alpha Vantage using the TIME_SERIES_DAILY endpoint.
    """
    params = {
        "function": "TIME_SERIES_DAILY",
        "symbol": symbol,
        "apikey": api_token,
        "outputsize": "compact"  # "compact" returns the last 100 data points
    }
    try:
        response = requests.get(base_url, params=params)
        response.raise_for_status()
        data = response.json()
        if "Time Series (Daily)" not in data:
            print(f"No valid data for {symbol}: {data.get('Note') or data.get('Error Message')}")
            return pd.DataFrame()
        time_series = data["Time Series (Daily)"]
        # Convert the time series dict into a DataFrame with date and closing price.
        df = pd.DataFrame([
            {"date": pd.to_datetime(date), "stock_price": float(values["4. close"]), "ticker": symbol}
            for date, values in time_series.items()
        ])
        df = df.sort_values(by="date")
        return df
    except Exception as e:
        print(f"Error fetching data for {symbol}: {e}")
        return pd.DataFrame()

def get_top_companies(df, top_n=5):
    # For demonstration, we select the top companies by average stock price.
    avg_price = df.groupby('ticker')['stock_price'].mean().abs()
    top_companies = avg_price.sort_values(ascending=False).head(top_n).index.tolist()
    return top_companies

def extract_peaks_from_series(dates, values, num_peaks=3):
    # Find all local peaks in the stock price series.
    peaks_indices, _ = find_peaks(values)
    
    if len(peaks_indices) == 0:
        return []

    # Sort the peaks by price in descending order and select the top peaks.
    peak_values = values[peaks_indices]
    sorted_peak_indices = peaks_indices[np.argsort(peak_values)[::-1]]
    top_peak_indices = sorted_peak_indices[:num_peaks]
    
    return [(dates[i], values[i]) for i in top_peak_indices]

def main():
    load_dotenv()
    api_token = os.getenv('API_TOKEN')
    base_url = os.getenv('BASE_URL', 'https://www.alphavantage.co/query')
    symbols_str = os.getenv('SYMBOLS', '')
    if not symbols_str:
        print("No symbols provided in .env. Please add a SYMBOLS variable.")
        return
    symbols = [s.strip() for s in symbols_str.split(',') if s.strip()]
    
    all_data = []
    for symbol in symbols:
        print(f"Fetching data for {symbol}...")
        df_symbol = fetch_data_for_symbol(symbol, base_url, api_token)
        if not df_symbol.empty:
            all_data.append(df_symbol)
    if not all_data:
        print("No data fetched from API. Exiting.")
        return
    data_df = pd.concat(all_data, ignore_index=True)
    
    # Select the top companies based on average stock price.
    top_companies = get_top_companies(data_df, top_n=5)
    print("Top companies by average stock price:", top_companies)
    
    results = {}
    for company in top_companies:
        company_df = data_df[data_df['ticker'] == company].sort_values(by='date')
        dates = company_df['date'].values
        prices = company_df['stock_price'].values
        peaks = extract_peaks_from_series(dates, prices, num_peaks=3)
        results[company] = peaks

    # Display the results.
    for company, peaks in results.items():
        print(f"\nCompany: {company}")
        if peaks:
            for date, value in peaks:
                print(f"  Peak at {date}: {value}")
        else:
            print("  No peaks found.")

if __name__ == '__main__':
    main()
