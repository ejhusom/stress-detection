#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Preprocess raw WESAD data.

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
import os
import pickle
import sys

import numpy as np
import pandas as pd
import yaml

from config import DATA_PATH_RAW
from preprocess_utils import find_files

def preprocess(dir_path):
    """Preprocess WESAD data.

    Args:
        dir_path (str): Path to directory containing files.

    """

    filepaths = find_files(dir_path, file_extension=".pkl")

    dfs = []

    # for subject_number in subject_numbers:
    for filepath in filepaths:

        with open(filepath, "rb") as f:
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
        labels_to_keep = [1, 2, 3, 4]
        df = df[df["label"].isin(labels_to_keep)]
        df.label = df.label.replace({1: 0, 2: 1, 3: 0, 4: 0})
        df.reset_index(drop=True, inplace=True)

        print(f"Saved file {filepath}.")

        df.to_csv(
            DATA_PATH_RAW
            / (os.path.basename(filepath).replace(".pkl", "-preprocessed.csv"))
        )


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

    preprocess(sys.argv[1])
