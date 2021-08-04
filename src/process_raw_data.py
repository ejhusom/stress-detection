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
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pickle
import plotly.graph_objects as go

from preprocess_utils import split_sequences

pd.options.plotting.backend = "plotly"

def read_wesad_data(subject_number):

    data_file = f"assets/data/raw/WESAD/S{subject_number}.pkl"

    with open(data_file, 'rb') as f:
            data = pickle.load(f, encoding="latin1")

    label = data["label"]
    signal = data["signal"]
    chest = signal["chest"]
    wrist = signal["wrist"]

    chest_sample_freq = 700
    wrist_bvp_sample_freq = 64
    wrist_acc_sample_freq = 32
    wrist_eda_sample_freq = 4
    n_seconds = label.size/chest_sample_freq
    new_sample_freq = 256
    chest_timestamps = np.linspace(0, n_seconds, label.size)
    wrist_bvp_timestamps = np.linspace(0, n_seconds, wrist["BVP"].size)
    wrist_acc_timestamps = np.linspace(0, n_seconds, wrist["ACC"].shape[0])
    wrist_eda_timestamps = np.linspace(0, n_seconds, wrist["EDA"].size)
    new_timestamps = np.linspace(0, n_seconds, int(n_seconds*new_sample_freq))

    df = pd.DataFrame()

    df["label"] = label
    # df["chest_acc_x"] = chest["ACC"][:,0]
    # df["chest_acc_y"] = chest["ACC"][:,1]
    # df["chest_acc_z"] = chest["ACC"][:,2]
    df["chest_ecg"] = chest["ECG"]
    # df["chest_emg"] = chest["EMG"]
    # df["chest_eda"] = chest["EDA"]
    # df["chest_temp"] = chest["Temp"]
    # df["chest_resp"] = chest["Resp"]

    # fig = df.loc[0:3500,:].plot()
    # fig.write_html("plot.html")

    # df = reindex_data(df, chest_timestamps, new_timestamps)
    # df["wrist_bvp"] = reindex_data(wrist["BVP"].reshape(-1),
    #         wrist_bvp_timestamps, new_timestamps)

    # df["wrist_eda"] = reindex_data(wrist["EDA"].reshape(-1),
    #         wrist_eda_timestamps, new_timestamps)

    # df["wrist_temp"] = reindex_data(wrist["TEMP"].reshape(-1),
    #         wrist_eda_timestamps, new_timestamps)

    # for i, axis in enumerate(["x", "y", "z"]):
    #     df[f"wrist_acc_{axis}"] = reindex_data(wrist["ACC"][:,i],
    #             wrist_acc_timestamps, new_timestamps)

    # Remove unusable labels:
    labels_to_ignore = [0,5,6,7]
    df = df[~df["label"].isin(labels_to_ignore)]
    df.reset_index(drop=True, inplace=True)

    # df.index.name = "time"

    # fig = df.loc[0:5,:].plot()
    # fig.write_html("plot2.html")

    # fig = df["label"].plot()
    # fig.write_html("label.html")

    X, y = split_sequences(np.array(df), 256)

    # df.to_csv(f"{subject_number}.csv")
    # print(f"Saved subject number {subject_number}.")

    # return df

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

if __name__ == '__main__': 

    # subject_numbers = [2,3,4,5,6,7,8,9,10,11,13,14,15,16,17]
    # subject_numbers = [2,3,4,5]
    subject_numbers = [2]

    for s in subject_numbers:
        read_wesad_data(s)
