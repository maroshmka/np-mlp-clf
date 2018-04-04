import numpy as np

hid_activations = ['relu', 'logsig', 'tanh']
out_activations = ['linear', 'softmax', 'logsig']


## general activations

def relu(X):
    return np.where(X > 0, X, 0)


def drelu(X):
    return np.where(X > 0, 1, 0)


def logsig(X):
    return 1 / (1 + np.exp(-X))


def dlogsig(X):
    return logsig(X) * (1 - logsig(X))


def tanh(X):
    return np.tanh(X)


def dtanh(X):
    return 1 / np.cosh(X) ** 2


## output-only activations

def linear(X):
    return X


def dlinear(X):
    return np.ones(X.shape)


def softmax(X):
    return (np.exp(X) / np.sum(np.exp(X)))


def dsoftmax(X):
    out = np.zeros(X.shape)
    out[np.argmax(X)] = np.exp(X[np.argmax(X)]) / np.sum(np.exp(X))
    return out


## cost functions

def MSE(D, Y):
    return np.sum((D - Y) ** 2, axis=0)


def CE(D, Y):
    return -np.sum(np.log(Y) * D, axis=0)

