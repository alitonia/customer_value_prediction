import numpy as np


def mae(y_true, y_pred):
    return float(np.mean(np.abs(y_true - y_pred)))


def wape(y_true, y_pred):
    return float(np.sum(np.abs(y_true - y_pred)) / np.sum(y_true))


def medape(y_true, y_pred):
    return float(np.median(np.abs((y_true - y_pred) / np.maximum(y_true, 1e-8))))