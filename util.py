import random

import matplotlib
import numpy as np
from sklearn.model_selection import KFold

matplotlib.use('Agg')  # todo: remove or change if not working


def augment(X):
    if X.ndim == 1:
        return np.concatenate((X, [1]))
    else:
        pad = np.ones((1, X.shape[1]))
        return np.concatenate((X, pad), axis=0)


def onehot_decode(X, axis):
    return np.argmax(X, axis=axis)


def onehot_encode(L, c):
    if isinstance(L, int):
        L = [L]
    n = len(L)
    out = np.zeros((c, n))
    out[L, range(n)] = 1
    return np.squeeze(out)


# normalize inputs

def normalize(x, axis=1):
    """
    By rows...
    x1 - 5 4 68 0
    x2 - 8 6 5 0
    """
    _avg = x.mean(axis=axis, keepdims=True)
    _std = x.std(axis=axis, keepdims=True)

    return (x - _avg) / _std


def load_data(path):
    """
    Load data, convert classes to ints, split inputs and labels.
    :param path: path to data
    :return:
    """
    letter_map = {
        'A': 0,
        'B': 1,
        'C': 2
    }
    convert_letter = lambda x: letter_map[x.decode('UTF-8')]

    data = np.loadtxt(path, skiprows=1, converters={2: convert_letter}).T
    inputs = data[:-1]
    labels = data[-1].astype(int)

    return inputs, labels


def split_train_test(inputs, labels, ratio=0.8):
    """
    Randomly shuffle dataset and split it to training and testing.
    :return: tuple with training/testing inputs/labels
    """
    count = inputs.shape[1]
    ind = np.arange(count)
    random.shuffle(ind)
    split = int(count * ratio)
    train_ind = ind[:split]
    test_ind = ind[split:]

    train_inputs = inputs[:, train_ind]
    train_labels = labels[train_ind]

    test_inputs = inputs[:, test_ind]
    test_labels = labels[test_ind]

    return train_inputs, train_labels, test_inputs, test_labels


def k_fold_cross_validation(clf, inputs, labels, n, verbosity):
    kf = KFold(n_splits=n)
    i = 1
    train_acc, train_rmse = [], []
    test_acc, test_rmse = [], []

    for train, validate in kf.split(inputs.T):
        train_fold_inputs, train_fold_labels = inputs[:, train], labels[train]
        validate_fold_inputs, validate_fold_labels = inputs[:, validate], labels[validate]

        trainCE, trainRE = clf.train(train_fold_inputs, train_fold_labels)
        testCE, testRE = clf.test(validate_fold_inputs, validate_fold_labels)

        if verbosity > 1:
            print('Fold n.{}: CE = {:6.2%}, RE = {:.5f}'.format(i, testCE, testRE))

        train_acc.append(trainCE)
        train_rmse.append(trainRE)
        test_acc.append(testCE)
        test_rmse.append(testRE)
        i += 1

        # reset weights on classifier for evaluating next fold
        clf.init_weights()

    if verbosity > 0:
        print('After {n}-fold cross-validation'.format(n=n))
        print('CEs - AVG - {avg:.5f}, STD - {std:.5f}'.format(avg=np.mean(test_acc),
                                                              std=np.std(test_acc)))
        print('REs - AVG - {avg:.5f}, STD - {std:.5f}'.format(avg=np.mean(test_rmse),
                                                              std=np.std(test_rmse)))

    train_acc = np.mean(train_acc, axis=0)
    train_rmse = np.mean(train_rmse, axis=0)
    return list(train_acc), list(train_rmse), np.mean(test_acc), np.mean(test_rmse)


def save_confusion_matrix(true_labels, predicted_labels, n_classes):
    confusion_matrix = np.zeros((n_classes, n_classes))

    for g_true, predict in zip(true_labels, predicted_labels):
        confusion_matrix[g_true, predict] += 1

    with open('results/confusion.txt', 'w') as f:
        for row in confusion_matrix:
            f.write(str(row) + '\n')
