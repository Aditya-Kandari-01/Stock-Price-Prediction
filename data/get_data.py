import yfinance as yf
import pandas as pd
import ta

def fetch_stock_data(ticker="AAPL", period="2y", interval="1d"):
    df = yf.download(ticker, period=period, interval=interval)
    df = df[['Open', 'High', 'Low', 'Close', 'Volume']]
    df.dropna(inplace=True)

    close_prices = df['Close'].squeeze() 

    df['RSI'] = ta.momentum.RSIIndicator(close=close_prices).rsi()
    
    macd = ta.trend.MACD(close=close_prices)
    df['MACD'] = macd.macd()
    df['MACD_signal'] = macd.macd_signal()

    return df


if __name__ == "__main__":
    df = fetch_stock_data("AAPL")  
    print(df.tail())               
    df.to_csv("AAPL_with_indicators.csv") 
    print("\nSaved data with indicators to AAPL_with_indicators.csv")