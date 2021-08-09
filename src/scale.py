#!/usr/bin/env python3
"""Scaling the inputs of the data set.

Possible scaling methods

TODO:
    Implement scaling when there is only one workout file.

Author:   
    Erik Johannes Husom

Created:  
    2020-09-16

"""
import os
import sys

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
import yaml

from config import DATA_PATH, DATA_SCALED_PATH
from preprocess_utils import find_files, scale_data

def scale(dir_path):
    """Scale training and test data.

    Args:
        dir_path (str): Path to directory containing files.

    """

    filepaths = find_files(dir_path, file_extension=".csv")

    DATA_SCALED_PATH.mkdir(parents=True, exist_ok=True)

    params = yaml.safe_load(open("params.yaml"))["scale"]
    input_method = params["input"]
    output_method = params["output"]
    classification = yaml.safe_load(open("params.yaml"))["clean"]["classification"]
    onehot_encode_target = yaml.safe_load(open("params.yaml"))["clean"]["onehot_encode_target"]
    
    if input_method == "standard":
        scaler = StandardScaler()
    elif input_method == "minmax":
        scaler = MinMaxScaler()
    elif input_method == "robust":
        scaler = RobustScaler()
    elif input_method == None:
        scaler = StandardScaler()
    else:
        raise NotImplementedError(f"{input_method} not implemented.")

    if output_method == "standard":
        output_scaler = StandardScaler()
    elif output_method == "minmax":
        output_scaler = MinMaxScaler()
    elif output_method == "robust":
        output_scaler = RobustScaler()
    elif output_method == None:
        output_scaler = StandardScaler()
    else:
        raise NotImplementedError(f"{output_method} not implemented.")

    train_inputs = []
    train_outputs = []

    data_overview = {}

    output_columns = np.array(
            pd.read_csv(DATA_PATH / "output_columns.csv", index_col=0)
    ).reshape(-1)

    n_output_cols = len(output_columns)

    for filepath in filepaths:

        df = pd.read_csv(filepath, index_col=0)
        
        # Convert to numpy
        data = df.to_numpy()

        # Split into input (X) and output/target (y)
        # X = data[:, 1:].copy()
        # y = data[:, 0].copy().reshape(-1, 1)
        X = data[:, n_output_cols:].copy()
        y = data[:, 0:n_output_cols].copy()

        if not onehot_encode_target:
            y = y.reshape(-1, 1)

        if "train" in filepath:
            train_inputs.append(X)
            train_outputs.append(y)
            category = "train"
        elif "test" in filepath:
            category = "test"
        elif "calibrate" in filepath:
            category = "calibrate"
            
        data_overview[filepath] = {"X": X, "y": y, "category": category}

    X_train = np.concatenate(train_inputs)
    y_train = np.concatenate(train_outputs)

    # Fit a scaler to the training data
    scaler = scaler.fit(X_train)

    if not classification:
        output_scaler = output_scaler.fit(y_train)

    for filepath in data_overview:

        # Scale inputs
        if input_method == None:
            X=data_overview[filepath]["X"]
        else:
            X = scaler.transform(data_overview[filepath]["X"])

        # Scale outputs
        if output_method == None or classification:
            y = data_overview[filepath]["y"]
        else:
            y = output_scaler.transform(data_overview[filepath]["y"])

        # Save X and y into a binary file
        np.savez(
            DATA_SCALED_PATH
            / (
                os.path.basename(filepath).replace(
                    data_overview[filepath]["category"] + ".csv", 
                    data_overview[filepath]["category"] + "-scaled.npz"
                )
            ),
            X = X, 
            y = y
        )

if __name__ == "__main__":

    np.random.seed(2020)

    scale(sys.argv[1])

