#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Testing inference with TFLite model.

Author:
    Erik Johannes Husom

Created:
    2021-10-01

"""
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf

from sklearn.metrics import accuracy_score


df = pd.read_csv("data3.csv")

interpreter = tf.lite.Interpreter(
        model_path="model.tflite")
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

X = np.array(df.iloc[:, 5:8])
y = np.array(df.iloc[:,1])
y_hat = np.zeros_like(y) + 10

for i in range(len(y)):
# for i in range(10):
    input_data = np.array(X[i,:], dtype=np.float32).reshape([1,3])
    interpreter.set_tensor(input_details[0]["index"], input_data)
    interpreter.invoke()
    output_data = interpreter.get_tensor(output_details[0]["index"])
    results = np.squeeze(output_data)

    y_hat[i] = results


y_hat = pd.Series(y_hat)

# y_hat = y_hat.rolling(100).apply(lambda x: x.mode()[0])

# mean = y_hat.rolling(10000).mean()
# y_hat = np.where(mean < 0.5, 0, 1)

# y_hat = y_hat | mask

accuracy = accuracy_score(y, y_hat)

plt.figure()
plt.plot(y, label="true")
plt.plot(y_hat, "--", label="pred")
plt.title(f"Accuracy: {accuracy}")
# plt.plot(mask, alpha = 0.7)
plt.legend()
plt.show()
