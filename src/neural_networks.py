#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Creating deep learning model for estimating power from breathing.

Author:
    Erik Johannes Husom

Date:
    2020-09-16

"""
from tensorflow.keras import layers, models, optimizers
from tensorflow.random import set_seed


def cnn(
    input_x,
    input_y,
    output_length=1,
    kernel_size=2,
    output_activation="linear",
    loss="mse",
    metrics="mse",
):
    """Define a CNN model architecture using Keras.

    Args:
        input_x (int): Number of time steps to include in each sample, i.e. how
            much history is matched with a given target.
        input_y (int): Number of features for each time step in the input data.
        n_steps_out (int): Number of output steps.
        seed (int): Seed for random initialization of weights.
        kernel_size (int): Size of kernel in CNN.
        output_activation: Activation function for outputs.

    Returns:
        model (keras model): Model to be trained.

    """

    kernel_size = kernel_size

    model = models.Sequential()
    model.add(
        layers.Conv1D(
            filters=32,
            kernel_size=kernel_size,
            activation="relu",
            input_shape=(input_x, input_y),
            name="input_layer",
        )
    )
    # model.add(layers.MaxPooling1D(pool_size=4, name="pool_1"))
    # model.add(
    #     layers.Conv1D(
    #         filters=32, kernel_size=2, activation="relu", name="conv1d_2"
    #     )
    # )
    # model.add(
    #     layers.Conv1D(
    #         filters=64, kernel_size=2, activation="relu", name="conv1d_3"
    #     )
    # )
    # model.add(layers.MaxPooling1D(pool_size=4, name="pool_2"))
    model.add(layers.Dropout(0.5))
    # model.add(layers.Conv1D(filters=32, kernel_size=kernel_size,
    # activation="relu", name="conv1d_3"))
    model.add(layers.Flatten(name="flatten"))
    # model.add(layers.Dense(64, activation="relu", name="dense_2"))
    model.add(layers.Dense(32, activation="relu", name="dense_1"))
    # model.add(layers.Dense(64, activation="relu", name="dense_2"))
    # model.add(layers.Dense(32, activation="relu", name="dense_3"))
    model.add(
        layers.Dense(output_length, activation=output_activation, name="output_layer")
    )

    # opt = optimizers.Adam(lr=1e-2, beta_1=0.9, beta_2=0.999,
    #     epsilon=1e-8, decay=0.0001)

    # model.compile(optimizer=opt, loss=loss, metrics=metrics)
    model.compile(optimizer="adam", loss=loss, metrics=metrics)

    return model


def cnn2(
    input_x,
    input_y,
    output_length=1,
    kernel_size=2,
    output_activation="linear",
    loss="mse",
    metrics="mse",
):
    """Define a CNN model architecture using Keras.

    Args:
        input_x (int): Number of time steps to include in each sample, i.e. how
            much history is matched with a given target.
        input_y (int): Number of features for each time step in the input data.
        n_steps_out (int): Number of output steps.
        seed (int): Seed for random initialization of weights.
        kernel_size (int): Size of kernel in CNN.
        output_activation: Activation function for outputs.

    Returns:
        model (keras model): Model to be trained.

    """

    kernel_size = kernel_size

    model = models.Sequential()
    model.add(
        layers.Conv1D(
            filters=256,
            kernel_size=kernel_size,
            activation="relu",
            input_shape=(input_x, input_y),
            name="input_layer",
        )
    )
    # model.add(layers.MaxPooling1D(pool_size=4, name="pool_1"))
    model.add(
        layers.Conv1D(
            filters=128, kernel_size=kernel_size, activation="relu", name="conv1d_1"
        )
    )
    model.add(
        layers.Conv1D(
            filters=64, kernel_size=kernel_size, activation="relu", name="conv1d_2"
        )
    )
    model.add(layers.MaxPooling1D(pool_size=4, name="pool_1"))
    model.add(
        layers.Conv1D(
            filters=32, kernel_size=kernel_size, activation="relu", name="conv1d_3"
        )
    )
    # model.add(layers.Conv1D(filters=32, kernel_size=kernel_size,
    # activation="relu", name="conv1d_4"))
    # model.add(layers.Dropout(rate=0.1))
    model.add(layers.Flatten(name="flatten"))
    model.add(layers.Dense(128, activation="relu", name="dense_1"))
    model.add(layers.Dense(64, activation="relu", name="dense_2"))
    model.add(layers.Dense(32, activation="relu", name="dense_3"))
    # model.add(layers.Dropout(rate=0.1))
    model.add(
        layers.Dense(output_length, activation=output_activation, name="output_layer")
    )
    model.compile(optimizer="adam", loss=loss, metrics=metrics)

    # model.compile(optimizer=optimizers.Adam(lr=1e-8, beta_1=0.9, beta_2=0.999,
    #     epsilon=1e-8, decay=0.0001), loss=loss, metrics=metrics)

    return model


def dnn(
    input_x,
    output_length=1,
    seed=2020,
    output_activation="linear",
    loss="mse",
    metrics="mse",
):
    """Define a DNN model architecture using Keras.

    Args:
        input_x (int): Number of features.
        output_length (int): Number of output steps.
        output_activation: Activation function for outputs.

    Returns:
        model (keras model): Model to be trained.

    """

    set_seed(seed)

    model = models.Sequential()
    model.add(layers.Dense(4, activation="relu", input_dim=input_x))
    # model.add(layers.Dense(256, activation="relu"))
    # model.add(layers.Dense(128, activation="relu"))
    # model.add(layers.Dense(64, activation="relu"))
    # model.add(layers.Dense(4, activation="relu"))
    # model.add(layers.Dense(256, activation='relu', input_dim=input_x))
    # model.add(layers.Dropout(0.5))
    model.add(layers.Dense(4, activation="relu"))
    # model.add(layers.Dense(32, activation='relu'))
    # model.add(layers.Dense(32, activation='relu'))
    model.add(layers.Dense(output_length, activation=output_activation))

    # opt = optimizers.Adam(lr=1e-2, beta_1=0.9, beta_2=0.999,
    #     epsilon=1e-8, decay=0.0001)

    # model.compile(optimizer=opt, loss=loss, metrics=metrics)
    model.compile(optimizer="adam", loss=loss, metrics=metrics)

    return model


def lstm(
    window_size,
    n_features,
    n_steps_out=1,
    output_activation="linear",
    loss="mse",
    metrics="mse",
):
    """Define a LSTM model architecture using Keras.

    Args:
        window_size (int): Number of time steps to include in each sample, i.e.
            how much history should be matched with a given target.
        n_features (int): Number of features for each time step, in the input
            data.

    Returns:
        model (Keras model): Model to be trained.

    """

    model = models.Sequential()
    model.add(
        layers.LSTM(50, input_shape=(window_size, n_features)
            ))
    # , return_sequences=True, activation="relu"))
    # model.add(layers.Dropout(0.5))
    # model.add(layers.LSTM(50, activation='relu'))
    # model.add(layers.LSTM(16, activation='relu'))
    model.add(layers.Dense(128, activation="relu"))
    model.add(layers.Dense(n_steps_out, activation=output_activation))
    model.compile(optimizer="adam", loss=loss, metrics=metrics)

    return model

def lstm1(
    window_size,
    n_features,
    n_steps_out=1,
    output_activation="linear",
    loss="mse",
    metrics="mse",
):
    """Define a LSTM model architecture using Keras.

    Args:
        window_size (int): Number of time steps to include in each sample, i.e.
            how much history should be matched with a given target.
        n_features (int): Number of features for each time step, in the input
            data.

    Returns:
        model (Keras model): Model to be trained.

    """

    model = models.Sequential()
    model.add(
        layers.LSTM(15, input_shape=(window_size, n_features))
    )
    model.add(layers.Dense(n_steps_out, activation=output_activation))
    model.compile(optimizer="adam", loss=loss, metrics=metrics)

    return model


def cnndnn(input_x, input_y, n_forecast_hours, n_steps_out=1):
    """Define a model architecture that combines CNN and DNN.

    Parameters
    ----------
    input_x : int
        Number of time steps to include in each sample, i.e. how much history
        should be matched with a given target.
    input_y : int
        Number of features for each time step, in the input data.
    dense_x: int
        Number of features for the dense part of the network.
    n_steps_out : int
        Number of output steps.
    Returns
    -------
    model : Keras model
        Model to be trained.
    """
    kernel_size = 4

    input_hist = layers.Input(shape=(input_x, input_y))
    input_forecast = layers.Input(shape=((n_forecast_hours,)))

    c = layers.Conv1D(
        filters=64,
        kernel_size=kernel_size,
        activation="relu",
        input_shape=(input_x, input_y),
    )(input_hist)
    c = layers.Conv1D(filters=32, kernel_size=kernel_size, activation="relu")(c)
    c = layers.Flatten()(c)
    c = layers.Dense(128, activation="relu")(c)
    c = models.Model(inputs=input_hist, outputs=c)

    d = layers.Dense(256, input_dim=n_forecast_hours, activation="relu")(input_forecast)
    d = layers.Dense(128, activation="relu")(d)
    d = layers.Dense(64, activation="relu")(d)
    d = models.Model(inputs=input_forecast, outputs=d)

    combined = layers.concatenate([c.output, d.output])

    combined = layers.Dense(256, activation="relu")(combined)
    combined = layers.Dense(128, activation="relu")(combined)
    combined = layers.Dense(64, activation="relu")(combined)
    combined = layers.Dense(n_steps_out, activation="linear")(combined)

    model = models.Model(inputs=[c.input, d.input], outputs=combined)

    model.compile(optimizer="adam", loss="mae")

    return model


def model6(
    x_tr_feature_dim, y_tr_dim, lr_super=0.001, hidden_nodes=512, dropout=0.2, L2=0
):
    input_dimension = x_tr_feature_dim
    output_dimension = y_tr_dim
    n_steps, n_length, n_features = 10, 256, 6

    input_ = keras.Input((None, n_length, n_features))
    out = layers.TimeDistributed(
        layers.Conv1D(filters=64, kernel_size=3, strides=1, activation="relu")
    )(input_)
    out = layers.TimeDistributed(
        layers.Conv1D(filters=64, kernel_size=3, strides=1, activation="relu")
    )(out)
    out = layers.TimeDistributed(layers.Dropout(0.1))(out)
    out = layers.TimeDistributed(layers.MaxPooling1D(pool_size=2))(out)
    out = layers.TimeDistributed(layers.Flatten())(out)
    out = layers.LSTM(128, return_sequences=True)(out)
    out = layers.Dropout(0.2)(out)
    out = layers.LSTM(64, return_sequences=True)(out)
    out = layers.Dropout(0.2)(out)
    out = layers.LSTM(32)(out)
    out = layers.Dense(32, activation="relu")(out)
    # output_ = layers.Dense(output_dimension, activation='sigmoid')(out)
    output_ = layers.Dense(output_dimension, activation="softmax")(out)
    model = Model(inputs=input_, outputs=output_)
    op = keras.optimizers.Adam(lr=lr_super)
    # model.compile(loss='binary_crossentropy', optimizer=op, metrics=['accuracy'])
    model.compile(loss="categorical_crossentropy", optimizer=op, metrics=["accuracy"])
    model.summary()
    return model


def model4(
    x_tr_feature_dim, y_tr_dim, lr_super=0.001, hidden_nodes=512, dropout=0.5, L2=0
):
    input_dimension = x_tr_feature_dim
    output_dimension = y_tr_dim
    n_steps, n_length, n_features = 10, 256, 4

    input_ = keras.Input((None, n_length, n_features))
    out = layers.TimeDistributed(
        layers.Conv1D(filters=64, kernel_size=3, strides=1, activation="relu")
    )(input_)
    out = layers.TimeDistributed(
        layers.Conv1D(filters=64, kernel_size=3, strides=1, activation="relu")
    )(out)
    out = layers.TimeDistributed(layers.Dropout(dropout))(out)
    out = layers.TimeDistributed(layers.MaxPooling1D(pool_size=2))(out)
    out = layers.TimeDistributed(layers.Flatten())(out)
    out = layers.LSTM(100)(out)
    out = layers.Dropout(dropout)(out)
    out = layers.Dense(100, activation="relu")(out)
    output_ = layers.Dense(output_dimension, activation="softmax")(out)
    model = Model(inputs=input_, outputs=output_)

    op = keras.optimizers.Adam(lr=lr_super)
    model.compile(loss="categorical_crossentropy", optimizer=op, metrics=["accuracy"])
    model.summary()
    return model
