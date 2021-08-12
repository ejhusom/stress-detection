#!/usr/bin/env python3
"""Split data into training and test set.

Author:
    Erik Johannes Husom

Date:
    2021-02-24

"""
import os
import random
import sys

import numpy as np
import pandas as pd
import yaml

from config import DATA_SPLIT_PATH
from preprocess_utils import find_files


def split(dir_path):
    """Split data into train and test set.

    Training files and test files are saved to different folders.

    Args:
        dir_path (str): Path to directory containing files.

    """

    params = yaml.safe_load(open("params.yaml"))["split"]
    shuffle_files = params["shuffle_files"]
    shuffle_samples = params["shuffle_samples"]

    DATA_SPLIT_PATH.mkdir(parents=True, exist_ok=True)

    filepaths = find_files(dir_path, file_extension=".csv")

    # Handle special case where there is only one workout file.
    if isinstance(filepaths, str) or len(filepaths) == 1:
        filepath = filepaths[0]

        df = pd.read_csv(filepath, index_col=0)

        if shuffle_samples:
            df = df.sample(frac=1).reset_index(drop=True)

        # print(df)
        # X = np.array(df.iloc[:,4:].copy())
        # print(X.shape)
        # import seaborn as sns
        # xx = X.reshape(X.shape[0]//64, int(4*64))
        # xx = np.clip(xx, -5, 5)
        # xx = X[:,0,:]
        # xx = X.reshape(-1)

        # plt.imshow(xx, cmap="hot", interpolation='nearest')
        # import matplotlib.pyplot as plt
        # plt.plot(xx)
        # # plt.colorbar()
        # plt.show()
        # return 0

        train_size = int(len(df) * params["train_split"])

        # This is used when using conformal predictors.
        # It specifies the calibration set size.
        # Set to 0 in params.yml if no calibration is to be done.
        calibrate_size = int(len(df) * params["calibrate_split"])

        df_train = None
        df_test = None
        df_calibrate = None

        if params["calibrate_split"] == 0:
            df_train = df.iloc[:train_size]
            df_test = df.iloc[train_size:]
        else:
            df_train = df.iloc[:train_size]
            df_calibrate = df.iloc[train_size : train_size + calibrate_size]
            df_test = df.iloc[train_size + calibrate_size :]

        df_train.to_csv(
            DATA_SPLIT_PATH
            / (os.path.basename(filepath).replace("featurized", "train"))
        )

        df_test.to_csv(
            DATA_SPLIT_PATH / (os.path.basename(filepath).replace("featurized", "test"))
        )

        if params["calibrate_split"] != 0:
            df_calibrate.to_csv(
                DATA_SPLIT_PATH
                / (os.path.basename(filepath).replace("featurized", "calibrate"))
            )

    else:

        if shuffle_files:
            random.shuffle(filepaths)

        # Parameter 'train_split' is used to find out no. of files in training set
        file_split = int(len(filepaths) * params["train_split"])
        file_split_calibrate = int(len(filepaths) * params["calibrate_split"])

        training_files = []
        test_files = []
        calibrate_files = []

        if file_split_calibrate == 0:
            training_files = filepaths[:file_split]
            test_files = filepaths[file_split:]
        else:
            training_files = filepaths[:file_split]
            calibrate_files = filepaths[file_split : file_split + file_split_calibrate]
            test_files = filepaths[file_split + file_split_calibrate :]

        for filepath in filepaths:

            df = pd.read_csv(filepath, index_col=0)

            if shuffle_samples:
                df = df.sample(frac=1).reset_index(drop=True)

            if filepath in training_files:
                df.to_csv(
                    DATA_SPLIT_PATH
                    / (os.path.basename(filepath).replace("featurized", "train"))
                )
            elif filepath in test_files:
                df.to_csv(
                    DATA_SPLIT_PATH
                    / (os.path.basename(filepath).replace("featurized", "test"))
                )
            elif filepath in calibrate_files:
                df.to_csv(
                    DATA_SPLIT_PATH
                    / (os.path.basename(filepath).replace("featurized", "calibrate"))
                )


if __name__ == "__main__":

    np.random.seed(2029)

    split(sys.argv[1])
