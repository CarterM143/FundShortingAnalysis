import os
import datetime
import logging
import requests
import pandas as pd
import matplotlib.pyplot as plt
from dotenv import load_dotenv

# Setup logging to file and console.
logging.basicConfig(
    filename='insider_trading.log',
    level=logging.DEBUG,
    format='%(asctime)s %(levelname)s: %(message)s'
)
console = logging.StreamHandler()
console.setLevel(logging.DEBUG)
formatter = logging.Formatter('%(asctime)s %(levelname)s: %(message)s')
console.setFormatter(formatter)
logging.getLogger('').addHandler(console)

def check_api_limit(data, symbol, endpoint):
    """Check if the API response indicates that the daily limit is reached."""
    if isinstance(data, dict):
        # Check for any key that might indicate a limit has been reached.
        if "Note" in data or "Error Message" in data or "Information" in data:
            message = data.get("Note") or data.get("Error Message") or data.get("Information")
            logging.error(f"Daily API limit reached on {endpoint} for {symbol}: {message}")
            return True
    return False

def fetch_insider_transactions(symbol, api_key, base_url):
    """
    Fetch insider transactions for a given symbol using Alpha Vantage's INSIDER_TRANSACTIONS endpoint.
    Returns a list of transactions or None if the API limit is reached.
    """
    params = {
        "function": "INSIDER_TRANSACTIONS",
        "symbol": symbol,
        "apikey": api_key
    }
    try:
        response = requests.get(base_url, params=params)
        response.raise_for_status()
        data = response.json()
        logging.debug(f"Raw insider data for {symbol}: {data}")
        if check_api_limit(data, symbol, "INSIDER_TRANSACTIONS"):
            return None
        logging.info(f"Fetched insider transactions for {symbol}.")
        return data.get("data", [])
    except Exception as e:
        logging.error(f"Error fetching insider transactions for {symbol}: {e}")
        return []

def fetch_overview(symbol, api_key, base_url):
    """
    Fetch company overview for a given symbol using Alpha Vantage's OVERVIEW endpoint.
    Returns a dictionary of overview data or None if the API limit is reached.
    """
    params = {
        "function": "OVERVIEW",
        "symbol": symbol,
        "apikey": api_key
    }
    try:
        response = requests.get(base_url, params=params)
        response.raise_for_status()
        data = response.json()
        if check_api_limit(data, symbol, "OVERVIEW"):
            return None
        logging.info(f"Fetched overview for {symbol}.")
        return data
    except Exception as e:
        logging.error(f"Error fetching overview for {symbol}: {e}")
        return {}

def compute_insider_history(transactions, start_date, end_date):
    """
    Given a list of transactions, compute a daily time series of cumulative net insider value
    from start_date to end_date.
    
    Assumes each transaction record includes:
      - 'transactionDate' (YYYY-MM-DD),
      - 'transactionType' (containing "Buy" or "Sell"),
      - 'transactionValue' (dollar amount).
    Buys add and sells subtract.
    """
    records = []
    for txn in transactions:
        txn_date_str = txn.get("transactionDate", "")
        try:
            txn_date = datetime.datetime.strptime(txn_date_str, "%Y-%m-%d").date()
        except Exception:
            logging.warning(f"Skipping transaction with invalid date: {txn_date_str}")
            continue
        if txn_date < start_date or txn_date > end_date:
            continue
        try:
            value = float(txn.get("transactionValue", 0))
        except Exception:
            value = 0.0
        txn_type = txn.get("transactionType", "").lower()
        if "buy" in txn_type:
            net_value = value
        elif "sell" in txn_type:
            net_value = -value
        else:
            net_value = 0.0
        records.append({"date": txn_date, "net_value": net_value})
    
    if not records:
        return pd.DataFrame(columns=['date', 'net_insider'])
    
    df = pd.DataFrame(records)
    # Aggregate by day.
    daily = df.groupby('date')['net_value'].sum().reset_index()
    daily = daily.sort_values('date')
    
    # Create a complete daily date range.
    all_dates = pd.date_range(start=start_date, end=end_date)
    daily = daily.set_index('date').reindex(all_dates, fill_value=0).rename_axis('date').reset_index()
    # Compute cumulative net insider value.
    daily['net_insider'] = daily['net_value'].cumsum()
    return daily[['date', 'net_insider']]

def plot_insider_history(symbol, history_df, market_cap):
    """
    Plot the insider history. If market_cap is provided, plot net insider trading as a percentage
    of market cap. Otherwise, plot the raw cumulative net insider trading in dollars.
    """
    plt.figure(figsize=(10, 6))
    if market_cap is not None:
        # Calculate the ratio (percentage).
        history_df['ratio'] = history_df['net_insider'] / market_cap * 100
        plt.plot(history_df['date'], history_df['ratio'], marker='o', linestyle='-')
        plt.title(f"Net Insider Trading as % of Market Cap for {symbol}")
        plt.ylabel("Net Insider Trading (% of Market Cap)")
        filename = f"insider_history_{symbol}.png"
    else:
        # Plot raw net insider trading.
        plt.plot(history_df['date'], history_df['net_insider'], marker='o', linestyle='-')
        plt.title(f"Cumulative Net Insider Trading for {symbol}")
        plt.ylabel("Cumulative Net Insider Trading (USD)")
        filename = f"insider_history_{symbol}_raw.png"
    
    plt.xlabel("Date")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()
    logging.info(f"Saved plot for {symbol} as {filename}.")
    return filename

def main():
    load_dotenv()
    api_key = os.getenv("API_TOKEN")
    base_url = os.getenv("BASE_URL", "https://www.alphavantage.co/query")
    symbols_str = os.getenv("SYMBOLS", "")
    if not symbols_str:
        logging.error("No symbols provided in .env. Please add a SYMBOLS variable (comma-separated).")
        print("No symbols provided in .env. Please add a SYMBOLS variable (comma-separated).")
        return
    symbols = [s.strip() for s in symbols_str.split(",") if s.strip()]
    
    # Define the historical period: one year ago to today.
    end_date = datetime.date.today()
    start_date = end_date - datetime.timedelta(days=365)
    
    for symbol in symbols:
        logging.info(f"Processing {symbol}...")
        print(f"Processing {symbol}...")
        
        # Fetch insider transactions.
        transactions = fetch_insider_transactions(symbol, api_key, base_url)
        if transactions is None:
            logging.error("Daily API limit reached while fetching insider transactions. Stopping further processing.")
            print("Daily API limit reached while fetching insider transactions. Exiting script.")
            return  # Exit the script immediately.
        
        # Fetch company overview.
        overview = fetch_overview(symbol, api_key, base_url)
        if overview is None:
            logging.error("Daily API limit reached while fetching company overview. Stopping further processing.")
            print("Daily API limit reached while fetching company overview. Exiting script.")
            return  # Exit the script immediately.
        
        market_cap_str = overview.get("MarketCapitalization")
        if not market_cap_str:
            logging.warning(f"Market cap not available for {symbol}. Will plot raw net insider trading.")
            print(f"Market cap not available for {symbol}. Will plot raw net insider trading.")
            market_cap = None
        else:
            try:
                market_cap = float(market_cap_str)
            except Exception:
                logging.error(f"Invalid market cap for {symbol}: {market_cap_str}. Will plot raw net insider trading.")
                print(f"Invalid market cap for {symbol}. Will plot raw net insider trading.")
                market_cap = None
        
        history_df = compute_insider_history(transactions, start_date, end_date)
        if history_df.empty:
            logging.info(f"No insider transaction data for {symbol} in the past year.")
            print(f"No insider transaction data for {symbol} in the past year.")
            continue
        
        plot_insider_history(symbol, history_df, market_cap)
        logging.info(f"Finished processing {symbol}.")
        print(f"Finished processing {symbol}.")

if __name__ == "__main__":
    main()
