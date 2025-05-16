import streamlit as st
import yfinance as yf
import pandas as pd
import ta
import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error
from utils.lstm_utils import fetch_stock_data, load_lstm_model, predict_lstm_price, plot_actual_vs_predicted

@st.cache_resource
def load_model():
    return tf.keras.models.load_model("lstm_model.h5")

@st.cache_data
def fetch_data(ticker="AAPL", period="2y", interval="1d"):
    df = yf.download(ticker, period=period, interval=interval)
    df = df[['Open', 'High', 'Low', 'Close', 'Volume']].dropna()
    close_prices = df['Close'].squeeze()
    df['RSI'] = ta.momentum.RSIIndicator(close=close_prices).rsi()
    macd = ta.trend.MACD(close=close_prices)
    df['MACD'] = macd.macd()
    df['MACD_signal'] = macd.macd_signal()
    return df

def prepare_data(data, n_timesteps=60):
    features = []
    targets = []
    for i in range(n_timesteps, len(data)):
        features.append(data[i - n_timesteps:i])
        targets.append(data[i, 0])
    features = np.array(features)
    targets = np.array(targets)
    scaler = MinMaxScaler()
    features_flat = features.reshape(-1, 1)
    features_scaled = scaler.fit_transform(features_flat)
    features_scaled = features_scaled.reshape(features.shape)
    targets_scaled = scaler.transform(targets.reshape(-1, 1)).flatten()
    return features_scaled, targets_scaled, scaler

def plot_data(df):
    fig, ax = plt.subplots(3, 1, figsize=(12, 10), sharex=True)
    ax[0].plot(df.index, df['Close'], label='Close Price', color='blue')
    ax[0].set_title('Close Price')
    ax[0].legend()
    ax[1].plot(df.index, df['RSI'], label='RSI', color='orange')
    ax[1].axhline(70, color='red', linestyle='--')
    ax[1].axhline(30, color='green', linestyle='--')
    ax[1].set_title('Relative Strength Index (RSI)')
    ax[1].legend()
    ax[2].plot(df.index, df['MACD'], label='MACD', color='purple')
    ax[2].plot(df.index, df['MACD_signal'], label='MACD Signal', color='magenta')
    ax[2].set_title('MACD')
    ax[2].legend()
    plt.tight_layout()
    st.pyplot(fig)

def predict_price(df, days=5):
    df = df.reset_index()
    df['DateOrdinal'] = pd.to_datetime(df['Date']).map(pd.Timestamp.toordinal)
    X = df['DateOrdinal'].values.reshape(-1, 1)
    y = df['Close'].values
    model = LinearRegression()
    model.fit(X, y)
    last_date = df['DateOrdinal'].iloc[-1]
    future_dates = np.array([last_date + i for i in range(1, days+1)]).reshape(-1, 1)
    preds = model.predict(future_dates)
    future_dates = [pd.Timestamp.fromordinal(int(d)) for d in future_dates.flatten()]
    prediction_df = pd.DataFrame({
        'Date': future_dates,
        'Predicted Close': preds.flatten()
    })
    return prediction_df

def plot_actual_vs_predicted(df, prediction_df):
    fig, ax = plt.subplots(figsize=(12,6))
    ax.plot(df.index, df['Close'], label='Actual Close Price')
    ax.plot(prediction_df['Date'], prediction_df['Predicted Close'], label='Predicted Close Price', linestyle='--', marker='o')
    ax.set_title('Actual vs Predicted Close Price')
    ax.set_xlabel('Date')
    ax.set_ylabel('Price')
    ax.legend()
    st.pyplot(fig)

def evaluate_model(df, model, lookback=60):
    close_prices = df[['Close']].values
    X, y, scaler = prepare_data(close_prices, lookback)
    test_size = int(0.1 * len(X))
    X_test, y_test = X[-test_size:], y[-test_size:]
    y_pred = model.predict(X_test)
    y_true = scaler.inverse_transform(y_test.reshape(-1, 1)).flatten()
    y_pred = scaler.inverse_transform(y_pred).flatten()
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    return y_true, y_pred, mae, rmse

st.title("📈 Stock Market Analyzer")
ticker = st.text_input("Ticker Symbol", "AAPL")
df = fetch_stock_data(ticker)
st.subheader(f"Historical Close Price for {ticker}")
st.line_chart(df['Close'])
model = load_model()
future_days = st.slider("Days to Predict", 1, 30, 5)
lstm_pred_df = predict_lstm_price(df, model, future_days=future_days)
st.subheader(f"LSTM Predicted Close Price for Next {future_days} Days")
st.write(lstm_pred_df)
st.subheader("Actual vs LSTM Predicted Close Price")
plot_actual_vs_predicted(df, lstm_pred_df)

if st.checkbox("📊 Show Model Evaluation (MAE, RMSE, Chart)"):
    y_true, y_pred, mae, rmse = evaluate_model(df, model)
    st.write(f"📉 Mean Absolute Error: `{mae:.2f}`")
    st.write(f"📊 Root Mean Squared Error: `{rmse:.2f}`")
    result_df = pd.DataFrame({
        "Actual": y_true,
        "Predicted": y_pred
    })
    st.line_chart(result_df[-100:])
