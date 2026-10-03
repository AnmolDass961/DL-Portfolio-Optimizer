import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.optimizers import Adam

tf.random.set_seed(42)
def make_seq(returns, window=21):
    x, y = [], []
    for i in range(len(returns) - window):
        x.append(returns[i:i + window])
        y.append(returns[i + window])
    x = np.array(x).reshape(-1, window, 1)
    y = np.array(y)
    return x, y

def build_lstm(window=21, units=32, dropout=0.0, lr=0.001):
    model = Sequential([
        LSTM(units, input_shape=(window, 1)),
        Dropout(dropout),
        Dense(16, activation='relu'),
        Dense(1)
    ])
    model.compile(optimizer=Adam(learning_rate=lr), loss='mse')
    return model

def forecast_lstm_returns(train_data,epochs=5, window=21, units=32, dropout=0.0, lr=0.001):
    predictions = {}
    for ticker in train_data.columns:
        series = train_data[ticker].values
        x, y = make_seq(series, window=window)
        model = build_lstm(window=window, units=units, dropout=dropout, lr=lr)
        model.fit(x, y, batch_size=16, epochs=epochs, verbose=0)
        last_window = series[-window:].reshape(1, window, 1)
        next_prediction = model.predict(last_window, verbose=0)[0, 0]
        predictions[ticker] = next_prediction * 252
    return pd.Series(predictions)

