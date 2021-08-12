#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Train deep learning model to estimate power from breathing data.


Author:   
    Erik Johannes Husom

Created:  
    2020-09-16  

"""
import sys
import time

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import xgboost as xgb
import yaml
from joblib import dump, load
from sklearn.discriminant_analysis import (
    LinearDiscriminantAnalysis,
    QuadraticDiscriminantAnalysis,
)
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, roc_auc_score
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.utils import plot_model

from config import (
    DATA_PATH,
    MODELS_FILE_PATH,
    MODELS_PATH,
    PLOTS_PATH,
    TRAININGLOSS_PLOT_PATH,
)
from model import cnn, cnndnn, dnn, lstm, model4, model6


def train(filepath):
    """Train model to estimate power.

    Args:
        filepath (str): Path to training set.

    """

    MODELS_PATH.mkdir(parents=True, exist_ok=True)

    # Load parameters
    params = yaml.safe_load(open("params.yaml"))["train"]
    net = params["net"]
    use_early_stopping = params["early_stopping"]
    patience = params["patience"]
    classification = yaml.safe_load(open("params.yaml"))["clean"]["classification"]

    output_columns = np.array(
        pd.read_csv(DATA_PATH / "output_columns.csv", index_col=0)
    ).reshape(-1)

    n_output_cols = len(output_columns)

    # Load training set
    train = np.load(filepath)

    X_train = train["X"]
    y_train = train["y"]

    # import seaborn as sns
    # xx = X_train.reshape(X_train.shape[0], 10240)
    # xx = np.clip(xx, -5, 5)
    # ax = sns.heatmap(xx, linewidth=0.5)
    # plt.imshow(xx, cmap="hot", interpolation='nearest')
    # plt.colorbar()

    # print(X_train.shape)
    # plt.plot(X_train[49,:,0])
    # plt.imshow(x[0,:,:])
    # plt.show()

    n_features = X_train.shape[-1]

    hist_size = X_train.shape[-2]
    target_size = y_train.shape[-1]

    if classification:
        if len(np.unique(y_train)) > 2:
            output_activation = "softmax"
            loss = "categorical_crossentropy"
        else:
            output_activation = "sigmoid"
            loss = "binary_crossentropy"
        output_length = n_output_cols
        metrics = "accuracy"
        monitor_metric = "accuracy"
    else:
        output_activation = "linear"
        output_length = target_size
        loss = "mse"
        metrics = "mse"
        monitor_metric = "loss"

    # Build model
    if net == "cnn":
        print(X_train.shape)
        # X_train = np.reshape(X_train, (X_train.shape[0], 10, 256, 4))
        hist_size = X_train.shape[-2]
        # model = cnn(hist_size, n_features, output_length=output_length,
        #         kernel_size=params["kernel_size"],
        #         output_activation=output_activation, loss=loss, metrics=metrics
        # )
        model = model6(256, y_tr_dim=4)
    elif net == "dnn":
        model = dnn(
            n_features,
            output_length=output_length,
            output_activation=output_activation,
            loss=loss,
            metrics=metrics,
        )
    elif net == "dt":
        # model = DecisionTreeClassifier()
        # model = RandomForestClassifier(50)
        # model = xgb.XGBClassifier(n_estimators=300, max_depth=5)
        # model = xgb.XGBClassifier(n_estimators=800, max_depth=5)
        # model = LinearDiscriminantAnalysis()
        model = QuadraticDiscriminantAnalysis()
        # model = SVC()
    elif net == "lstm":
        hist_size = X_train.shape[-2]
        model = lstm(
            hist_size,
            n_features,
            n_steps_out=output_length,
            output_activation=output_activation,
            loss=loss,
            metrics=metrics,
        )
    else:
        raise NotImplementedError("Only 'cnn' is implemented.")

    if net == "dt":
        model.fit(X_train, y_train)
        dump(model, MODELS_FILE_PATH)
    else:
        print(model.summary())

        # Save a plot of the model. Will not work if Graphviz is not installed, and
        # is therefore skipped if an error is thrown.
        try:
            PLOTS_PATH.mkdir(parents=True, exist_ok=True)
            plot_model(
                model,
                to_file=PLOTS_PATH / "model.png",
                show_shapes=False,
                show_layer_names=True,
                rankdir="TB",
                expand_nested=True,
                dpi=96,
            )
        except:
            print("Failed saving plot of the network architecture.")

        early_stopping = EarlyStopping(
            monitor="val_" + monitor_metric, patience=patience, verbose=4
        )

        model_checkpoint = ModelCheckpoint(
            MODELS_FILE_PATH, monitor="val_" + monitor_metric, save_best_only=True
        )

        if use_early_stopping:
            # Train model for 10 epochs before adding early stopping
            history = model.fit(
                X_train,
                y_train,
                epochs=10,
                batch_size=params["batch_size"],
                validation_split=0.25,
            )

            loss = history.history[monitor_metric]
            val_loss = history.history["val_" + monitor_metric]

            history = model.fit(
                X_train,
                y_train,
                epochs=params["n_epochs"],
                batch_size=params["batch_size"],
                validation_split=0.25,
                callbacks=[early_stopping, model_checkpoint],
            )

            loss += history.history[monitor_metric]
            val_loss += history.history["val_" + monitor_metric]

        else:
            history = model.fit(
                X_train,
                y_train,
                epochs=params["n_epochs"],
                batch_size=params["batch_size"],
                validation_split=0.25,
            )

            loss = history.history["loss"]
            val_loss = history.history["val_loss"]

            model.save(MODELS_FILE_PATH)

        TRAININGLOSS_PLOT_PATH.parent.mkdir(parents=True, exist_ok=True)

        print(f"Best model in epoch: {np.argmax(np.array(val_loss))}")

        n_epochs = range(len(loss))

        plt.figure()
        plt.plot(n_epochs, loss, label="Training loss")
        plt.plot(n_epochs, val_loss, label="Validation loss")
        plt.legend()
        plt.savefig(TRAININGLOSS_PLOT_PATH)


if __name__ == "__main__":

    np.random.seed(2020)

    train(sys.argv[1])
