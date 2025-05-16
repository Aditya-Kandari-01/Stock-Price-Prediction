import numpy as np
import pandas as pd
import streamlit as st
import yfinance as yf
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
import matplotlib.pyplot as plt

@st.cache_data
def fetch_stock_data(ticker="AAPL", period="2y", interval="1d"):
    df = yf.download(ticker, period=period, interval=interval)
    df = df[['Close']].dropna()
    return df

def prepare_lstm_data(data, lookback=60):
    scaler = MinMaxScaler()
    scaled = scaler.fit_transform(data)
    X = []
    for i in range(lookback, len(scaled)):
        X.append(scaled[i-lookback:i])
    X = np.array(X)
    return X, scaler

def load_lstm_model(path="models/lstm_model.h5"):
    return tf.keras.models.load_model(path)

def predict_lstm_price(df, model, lookback=60, future_days=5):
    close_prices = df[['Close']].values
    X, scaler = prepare_lstm_data(close_prices, lookback)
    last_sequence = X[-1]
    preds = []
    current_seq = last_sequence
    for _ in range(future_days):
        current_seq_reshaped = current_seq.reshape(1, lookback, 1)
        pred = model.predict(current_seq_reshaped)[0][0]
        preds.append(pred)
        current_seq = np.append(current_seq[1:], [[pred]], axis=0)
    preds = np.array(preds).reshape(-1, 1)
    preds = scaler.inverse_transform(preds).flatten()
    last_date = df.index[-1]
    future_dates = pd.date_range(last_date + pd.Timedelta(days=1), periods=future_days)
    prediction_df = pd.DataFrame({
        'Date': future_dates,
        'Predicted Close': preds
    })
    return prediction_df

def plot_actual_vs_predicted(actual_df, predicted_df):
    plt.figure(figsize=(12,6))
    plt.plot(actual_df.index, actual_df['Close'], label='Actual Close Price')
    plt.plot(predicted_df['Date'], predicted_df['Predicted Close'], label='Predicted Close Price', linestyle='--', marker='o')
    plt.title('Actual vs Predicted Close Price')
    plt.xlabel('Date')
    plt.ylabel('Price')
    plt.legend()
    plt.grid(True)
    st.pyplot(plt.gcf())
    plt.clf()

st.title("📈 AAPL Stock Price Prediction Using LSTM")
df = fetch_stock_data("AAPL")
st.subheader("Historical Close Price")
st.line_chart(df['Close'])
model = load_lstm_model("models/lstm_model.h5")
st.subheader("LSTM Predicted Close Price for Next 5 Days")
lstm_prediction_df = predict_lstm_price(df, model)
st.write(lstm_prediction_df)
st.subheader("Actual vs LSTM Predicted Close Price")
plot_actual_vs_predicted(df, lstm_prediction_df)
