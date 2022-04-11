from __future__ import annotations
from keras.models import Sequential
from keras.layers import (
    Flatten,
    Dense,
    LSTM,
    TimeDistributed,
    ConvLSTM2D,
    BatchNormalization,
    Dropout,
)
from keras.layers.convolutional import Conv1D, MaxPooling1D
import numpy as np
import tensorflow as tf
import math


def convModel(
    n_conv_layers: int,
    n_conv_filters: int,
    kernel_size: int,
    input_shape: tuple,
    n_dense_nodes: int,
    n_output_nodes: int,
):
    model = Sequential()
    for nLayer in range(n_conv_layers):
        if nLayer == 0:
            model.add(
                Conv1D(
                    n_conv_filters,
                    kernel_size=kernel_size,
                    activation="relu",
                    input_shape=input_shape,
                )
            )
        else:
            model.add(
                Conv1D(n_conv_filters, kernel_size=kernel_size, activation="relu")
            )
    model.add(MaxPooling1D())
    model.add(Flatten())
    model.add(Dense(n_dense_nodes, activation="relu"))
    model.add(Dense(n_output_nodes))
    return model


def naiveModel(train_y: np.array):

    """
    A naive model to baseline. For each step, the value of the previous step is used as the prediction
    """
    predictions = []
    for i in range(1, len(train_y)):
        predictions.append(train_y[i - 1])

    return predictions


def mlpModel(
    n_hidden_layers: int, n_dense_nodes: int, n_output_nodes: int, n_steps: int
):

    model = Sequential()
    for nLayer in range(n_hidden_layers):
        if nLayer == 0:
            model.add(Dense(n_dense_nodes, activation="relu", input_dim=n_steps))
        else:
            model.add(
                Dense(math.floor(n_dense_nodes / (nLayer + 1)), activation="relu")
            )
    model.add(Dense(n_output_nodes))
    return model


def lstmModel(
    n_lstm_layers: int,
    n_lstm_units: int,
    n_dense_nodes: int,
    n_output_nodes: int,
    input_shape: tuple,
):
    model = Sequential()
    timesteps, features = input_shape
    for nLayer in range(n_lstm_layers):
        if nLayer == 0:
            model.add(
                LSTM(
                    n_lstm_units,
                    activation="relu",
                    return_sequences=True,
                    input_shape=(timesteps, features),
                )
            )
        else:
            model.add(LSTM(n_lstm_units, activation="relu"))

    model.add(Dense(n_dense_nodes, activation="relu"))
    model.add(Dense(n_output_nodes))

    return model


def cnnLSTMModel(
    n_conv_layers: int,
    n_conv_filters: int,
    kernel_size: int,
    n_lstm_units: int,
    n_dense_nodes: int,
    n_output_nodes: int,
    input_shape: tuple,
):

    model = Sequential()
    for nLayer in range(n_conv_layers):
        if nLayer == 0:
            model.add(
                TimeDistributed(
                    Conv1D(
                        n_conv_filters,
                        kernel_size=kernel_size,
                        activation="relu",
                        input_shape=input_shape,
                    )
                )
            )
            # model.add(TimeDistributed(MaxPooling1D()))
            model.add(Dropout(0.2))
        else:
            model.add(
                TimeDistributed(
                    Conv1D(
                        n_conv_filters, kernel_size=kernel_size // 2, activation="relu"
                    )
                )
            )
            model.add(TimeDistributed(MaxPooling1D()))
            model.add(Dropout(0.2))
            model.add(TimeDistributed(Flatten()))

    model.add(LSTM(n_lstm_units, activation="relu", return_sequences=True))
    model.add(Dropout(0.2))
    model.add(LSTM(n_lstm_units * 2, activation="relu"))
    model.add(Dropout(0.2))

    # model.add(Dense(n_dense_nodes, activation="relu"))
    model.add(Dense(n_output_nodes))

    return model


def convLSTMModel(
    n_conv_filters: int,
    n_rows: int,
    n_cols: int,
    n_dense_nodes: int,
    n_output_nodes: int,
    input_shape: tuple,
):
    model = Sequential()

    model.add(
        ConvLSTM2D(
            n_conv_filters,
            (n_rows, n_cols),
            activation="relu",
            return_sequences=True,
            input_shape=input_shape,
            padding="same",
        )
    )
    model.add(BatchNormalization())
    model.add(ConvLSTM2D(n_conv_filters * 2, (n_rows, n_cols), activation="relu"))
    model.add(BatchNormalization())
    # model.add(ConvLSTM2D(n_conv_filters, (n_rows, n_cols), activation="relu"))
    # model.add(BatchNormalization())
    # model.add(ConvLSTM2D(n_conv_filters, (n_rows, n_cols), activation="relu")),
    model.add(Flatten()),
    model.add(Dense(n_dense_nodes, activation="relu")),
    model.add(Dense(n_output_nodes))

    return model
