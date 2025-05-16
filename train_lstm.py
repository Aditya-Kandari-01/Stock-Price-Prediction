import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__))))
from data.get_data import fetch_stock_data
from models.lstm_model import prepare_data, build_lstm_model
import numpy as np
import tensorflow as tf


df = fetch_stock_data("AAPL")
close_prices = df[['Close']].values


X, y, scaler = prepare_data(close_prices)
X = X.reshape((X.shape[0], X.shape[1], 1))


model = build_lstm_model((X.shape[1], 1))
model.fit(X, y, epochs=10, batch_size=32)


model.save("lstm_model.h5")
print("✅ LSTM model trained and saved as lstm_model.h5")