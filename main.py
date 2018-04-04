import argparse
import json
import time

from classifier import *
from grid_search import GridSearch
from visualize import *


def load_iris():
    # load data
    data = np.loadtxt('iris.dat').T
    inputs = data[:-1]
    labels = data[-1].astype(int) - 1

    # normalize inputs
    inputs = normalize(inputs)

    return split_train_test(inputs, labels, ratio=0.8)


def load_2d():
    # load data
    train_inputs, train_labels = load_data('2d.trn.dat')
    test_inputs, test_labels = load_data('2d.tst.dat')

    # normalize inputs
    train_inputs = normalize(train_inputs)
    test_inputs = normalize(test_inputs)

    return train_inputs, train_labels, test_inputs, test_labels


def select_model(train_inputs, train_labels, dim, n_classes):
    grid = {
        'eps': [200, 500],
        'mini_batch_size': [1, 12],
        'dim_hid': [20, 30],
        'alpha': [0.1, 0.3],
        'out_function': ['linear', 'logsig'],
        'hid_function': ['logsig', 'tanh'],
    }

    search = GridSearch(clf=MLPClassifier, grid=grid, dim=dim, n_classes=n_classes)
    search.do_search(inputs=train_inputs, labels=train_labels)
    best_params = search.get_best_params()

    print('Best params - {bp}. Saving...'.format(bp=best_params))
    with open('results/best_model_params', 'w') as f:
        json.dump(best_params, f)

    search.save_report()
    best_params = best_params['params']
    return best_params


def run(train_inputs, train_labels, test_inputs, test_labels, grid_search):
    dim, count = train_inputs.shape
    n_classes = np.max(train_labels) + 1

    #
    # model selection
    # do grid search and pick best model
    #
    params = None
    if grid_search:
        params = select_model(train_inputs, train_labels, dim, n_classes)
    else:
        params = {
            'eps': 30,
            'mini_batch_size': 12,
            'dim_hid': 30,
            'alpha': 0.1,
            'out_function': 'linear',
            'hid_function': 'logsig',
        }

    #
    # # final test
    #
    started = time.time()
    print('Started testing model on test data...')

    print('Training best model...')
    clf = MLPClassifier(dim_in=dim, n_classes=n_classes, **params)
    trainCE, trainRE = clf.train(train_inputs, train_labels, verbosity=2)
    testCE, testRE = clf.test(test_inputs, test_labels)
    predicted = clf.predict(test_inputs)

    print('Plotting erros...')
    plot_both_errors(trainCE, trainRE, testCE, testRE)

    print('Plotting dots...')
    plot_dots(test_inputs, test_labels, predicted)

    print('Computing confusion matrix...')
    save_confusion_matrix(test_labels, predicted, n_classes)

    print('Done in {t:.3f}s...'.format(t=time.time() - started))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--data",
                        type=str,
                        required=True,
                        choices=['iris', '2d'],
                        help="Pick data to load. Choices - iris/2d"
                        )
    parser.add_argument("-gs",
                        default=False,
                        required=False,
                        action='store_true',
                        help="Run with grid search."
                        )
    args = parser.parse_args()

    load_method = load_iris if args.data == 'iris' else load_2d
    train_inputs, train_labels, test_inputs, test_labels = load_method()
    
    run(train_inputs, train_labels, test_inputs, test_labels, args.gs)
