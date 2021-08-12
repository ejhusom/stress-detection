#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Explore WESAD.

Author:   
    Erik Johannes Husom

Created:  
    2021-07-05


Notes:

    Labels:

    0: N/A
    1: Baseline
    2: Stress
    3: Amusement
    4: Meditation
    5/6/7: Should be ignored

"""
import json
import pickle

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, RobustScaler, StandardScaler

from config import DATA_PATH, MODELS_FILE_PATH, MODELS_PATH, TRAININGLOSS_PLOT_PATH
from model import cnn, lstm, model4, model6
from preprocess_utils import move_column, split_sequences

pd.options.plotting.backend = "plotly"


def preprocess_wesad_data(subject_numbers):

    dfs = []

    for subject_number in subject_numbers:

        data_file = f"assets/data/raw/WESAD/S{subject_number}.pkl"

        with open(data_file, "rb") as f:
            data = pickle.load(f, encoding="latin1")

        label = data["label"]
        signal = data["signal"]
        chest = signal["chest"]
        wrist = signal["wrist"]

        chest_sample_freq = 700
        # wrist_bvp_sample_freq = 64
        # wrist_acc_sample_freq = 32
        # wrist_eda_sample_freq = 4
        n_seconds = label.size / chest_sample_freq
        new_sample_freq = 64
        label_timestamps = np.linspace(0, n_seconds, label.size)
        wrist_bvp_timestamps = np.linspace(0, n_seconds, wrist["BVP"].size)
        wrist_acc_timestamps = np.linspace(0, n_seconds, wrist["ACC"].shape[0])
        wrist_eda_timestamps = np.linspace(0, n_seconds, wrist["EDA"].size)
        new_timestamps = np.linspace(0, n_seconds, int(n_seconds * new_sample_freq))

        df = pd.DataFrame()

        df["label"] = label
        # df["chest_acc_x"] = chest["ACC"][:,0]
        # df["chest_acc_y"] = chest["ACC"][:,1]
        # df["chest_acc_z"] = chest["ACC"][:,2]
        # df["chest_ecg"] = chest["ECG"]
        # df["chest_emg"] = chest["EMG"]
        # df["chest_eda"] = chest["EDA"]
        # df["chest_temp"] = chest["Temp"]
        # df["chest_resp"] = chest["Resp"]

        # fig = df.loc[0:3500,:].plot()
        # fig.write_html("plot.html")

        df = reindex_data(df, label_timestamps, new_timestamps)
        df["wrist_bvp"] = wrist["BVP"]
        # df["wrist_bvp"] = reindex_data(wrist["BVP"].reshape(-1),
        #         wrist_bvp_timestamps, new_timestamps)

        df["wrist_eda"] = reindex_data(
            wrist["EDA"].reshape(-1), wrist_eda_timestamps, new_timestamps
        )

        df["wrist_temp"] = reindex_data(
            wrist["TEMP"].reshape(-1), wrist_eda_timestamps, new_timestamps
        )

        for i, axis in enumerate(["x", "y", "z"]):
            df[f"wrist_acc_{axis}"] = reindex_data(
                wrist["ACC"][:, i], wrist_acc_timestamps, new_timestamps
            )

        # Remove unusable labels:
        labels_to_ignore = [0, 4, 5, 6, 7]
        df = df[~df["label"].isin(labels_to_ignore)]
        # df.label = df.label.replace({1: 0, 2: 1, 3: 0, 4: 0})
        df.reset_index(drop=True, inplace=True)

        print(f"Saved subject number {subject_number}.")

        df.to_csv(f"assets/data/raw/wesad_csv/{subject_number}.csv")
        dfs.append(df)

    df = pd.concat(dfs, ignore_index=True)

    label = np.array(df["label"].copy())

    del df["label"]
    df = StandardScaler().fit_transform(np.array(df))
    df = pd.DataFrame(df)
    df["label"] = label

    df = move_column(df, "label", 0)

    # df.index.name = "time"

    # fig = df.loc[0:5,:].plot()
    # fig.write_html("plot2.html")

    # fig = df["label"].plot()
    # fig.write_html("label.html")
    # print(df)

    X, y = split_sequences(np.array(df), 1)
    # X = np.array(df.iloc[:,1:])
    # y = np.array(df.iloc[:,0])

    # print(y)
    np.savez("assets/data/wesad.npz", X=X, y=y)


def train():

    data = np.load("assets/data/wesad.npz")
    X = data["X"]
    y = data["y"] - 1

    # xx = X.reshape(X.shape[0]//64, int(4*64))
    # # xx = np.clip(xx, -5, 5)
    # plt.imshow(xx, cmap="hot", interpolation='nearest')
    # plt.colorbar()

    print(X.shape)
    # plt.plot(X[49,:,0])
    # plt.imshow(x[0,:,:])
    # plt.show()

    # X = np.flip(X, 1)

    # xx = X.flatten()
    # import matplotlib.pyplot as plt
    # # plt.plot(X[0,:])
    # plt.plot(xx)
    # plt.show()
    # X = X.flatten()

    # X = np.reshape(X, (X.shape[0], 10, 256, 6))

    # import matplotlib.pyplot as plt
    # plt.plot(X[0,:,0])
    # plt.show()

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=5, shuffle=False
    )

    y_train = tf.keras.utils.to_categorical(y_train, num_classes=3)
    y_test = tf.keras.utils.to_categorical(y_test, num_classes=3)

    np.savez("assets/data/combined/test.npz", X=X_test, y=y_test)

    # model = model6(256, y_tr_dim=3)
    model = lstm(
        X_train.shape[-2],
        X_train.shape[-1],
        3,
        output_activation="softmax",
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )

    model.fit(
        X_train, y_train, epochs=20, batch_size=64, validation_split=0.2, shuffle=True
    )

    model.save(MODELS_FILE_PATH)


def reindex_data(data, old_timestamps, new_timestamps, method="nearest"):
    """Reindex data by using timestamps as reference.

    Args:
        data (array-like): An array, list, DataFrame or similar containing
            data to be reindexed.
        old_timestamps (array-like): The timestamps corresponding to each of
            the elements in 'data'.
        new_timestamps (array-like): The new set of timestamps to which the
            data shall be reindexed to.
        method (str): Which method to user for padding the data.

    Returns:
        df (DataFrame): A DataFrame containing the reindexed data.

    """

    df = pd.DataFrame(data)
    df = df.set_index(old_timestamps)
    df = df.reindex(new_timestamps, method=method)

    return df


if __name__ == "__main__":

    subject_numbers = [2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 13, 14, 15, 16, 17]
    # subject_numbers = [2,3,4,5]
    # subject_numbers = [2]
    # preprocess_wesad_data(subject_numbers)

    train()
