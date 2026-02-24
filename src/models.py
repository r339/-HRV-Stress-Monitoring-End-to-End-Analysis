"""
hrv_stress_monitoring/src/models.py
-------------------------------------
LSTM model builder for binary HRV stress classification.

Architecture
------------
  Input  → LSTM(64) → Dropout(0.3) → BatchNorm
         → LSTM(32) → Dropout(0.3)
         → Dense(16, relu) → Dense(1, sigmoid)
"""

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (
    LSTM, Dense, Dropout, BatchNormalization, Input
)
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam


def build_lstm_model(n_timesteps: int,
                     n_features: int,
                     lstm_units_1: int = 64,
                     lstm_units_2: int = 32,
                     dropout_rate: float = 0.3,
                     learning_rate: float = 1e-3) -> tf.keras.Model:
    """
    Build and compile the 2-layer LSTM stress classifier.

    Parameters
    ----------
    n_timesteps : int
        Number of time steps (window size in beats, e.g. 60).
    n_features : int
        Number of input features per step (e.g. 12).
    lstm_units_1 : int
        Units in the first LSTM layer.
    lstm_units_2 : int
        Units in the second LSTM layer.
    dropout_rate : float
        Dropout fraction after each LSTM layer.
    learning_rate : float
        Adam learning rate.

    Returns
    -------
    tf.keras.Model
        Compiled Keras model.
    """
    model = Sequential([
        Input(shape=(n_timesteps, n_features)),
        LSTM(lstm_units_1, return_sequences=True),
        Dropout(dropout_rate),
        BatchNormalization(),
        LSTM(lstm_units_2, return_sequences=False),
        Dropout(dropout_rate),
        Dense(16, activation='relu'),
        Dense(1, activation='sigmoid'),
    ], name='hrv_lstm_classifier')

    model.compile(
        optimizer=Adam(learning_rate=learning_rate),
        loss='binary_crossentropy',
        metrics=['accuracy', tf.keras.metrics.AUC(name='auc')],
    )
    return model


def get_callbacks(patience_es: int = 10,
                  patience_lr: int = 5,
                  min_lr: float = 1e-6) -> list:
    """
    Return standard training callbacks.

    Parameters
    ----------
    patience_es : int
        EarlyStopping patience (epochs).
    patience_lr : int
        ReduceLROnPlateau patience.
    min_lr : float
        Minimum learning rate for plateau scheduler.

    Returns
    -------
    list[tf.keras.callbacks.Callback]
    """
    return [
        EarlyStopping(
            monitor='val_loss', patience=patience_es,
            restore_best_weights=True, verbose=1
        ),
        ReduceLROnPlateau(
            monitor='val_loss', factor=0.5, patience=patience_lr,
            min_lr=min_lr, verbose=1
        ),
    ]
