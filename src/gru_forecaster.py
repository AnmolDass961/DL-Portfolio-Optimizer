import numpy as np
import pandas as pd
import tensorflow as tf
from src.lstm_forecaster import make_seq
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout,GRU
from tensorflow.keras.optimizers import Adam
tf.random.set_seed(42)

def build_gru(window=21, units=32, dropout=0.0, lr=0.001):
    model = Sequential([
        GRU(units, input_shape=(window, 1)),
        Dropout(dropout),
        Dense(16, activation='relu'),
        Dense(1)
    ])
    model.compile(optimizer=Adam(learning_rate=lr), loss='mse')
    return model

def forecast_gru_returns(train_data,epochs=5, window=21, units=32, dropout=0.0, lr=0.001):
    predictions = {}
    for ticker in train_data.columns:
        series = train_data[ticker].values
        x, y = make_seq(series, window=window)
        model = build_gru(window=window, units=units, dropout=dropout, lr=lr)
        model.fit(x, y, batch_size=16,epochs=epochs, verbose=0)
        last_window = series[-window:].reshape(1, window, 1)
        next_prediction = model.predict(last_window, verbose=0)[0, 0]
        predictions[ticker] = next_prediction * 252
    return pd.Series(predictions)