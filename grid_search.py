import asyncio
import concurrent.futures
import itertools

import pandas as pd

from util import k_fold_cross_validation, split_train_test


class GridSearch:
    def __init__(self, clf, grid, dim, n_classes):
        self.clf = clf
        self.grid = grid
        self.dim = dim
        self.n_classes = n_classes
        self.results = None

    def _make_opts_combinations(self):
        """
        Take the grid and make all combinations of paramters from it.
        :return: list of dicts
        """
        for k, v in self.grid.items():
            self.grid[k] = [(k, vv) for vv in v]

        grid_list = [v for k, v in self.grid.items()]
        input_combinations = list(itertools.product(*grid_list))

        input_combinations_list_dict = []
        required_fields = {'dim_in': self.dim, 'n_classes': self.n_classes + 1}

        # make dicts for kwargs from combinations

        for i, combination in enumerate(input_combinations):
            opts = required_fields.copy()
            comb_dict = {c[0]: c[1] for c in combination}
            opts.update(comb_dict)
            input_combinations_list_dict.append(opts)

        return input_combinations_list_dict

    def train_clf(self, opts, inputs, labels, i):
        """
        Train clf with parameters.
        :param opts: dict - parameters to init clf.
        :param inputs: np.array of inputs. column oriented
        :param labels: np.array of labels
        :return:
        """
        print('{i}. Starting validate model with parameters - {params}'.format(i=i, params=opts))
        clf = self.clf(**opts)
        train_inputs, train_labels, test_inputs, test_labels = split_train_test(inputs, labels)    
        trainCE, trainRE = clf.train(train_inputs, train_labels)
        testCE, testRE = clf.test(test_inputs, test_labels)
        print('{i}. Done'.format(i=i))

        score = {
            'acc': testCE,
            'rmse': testRE
        }
        params = clf.get_params()
        result = {
            'score': score,
            'params': params
        }
        print('{i} - results - {r}'.format(i=i, r=result))
        return result

    def do_search(self, inputs, labels):
        """
        Do grid search through all parameters combinations.
        :param inputs: np.array of inputs. column oriented
        :param labels: np.array of labels
        :return: results - list of dicts
            {
                'score': {'acc':x, 'rmse':y}
                'params': {<model_params>}
            }
        """
        input_combinations_list_dict = self._make_opts_combinations()
        print('Length of inputs combination - {l}'.format(l=len(input_combinations_list_dict)))

        results = []
        for i, input_combination in enumerate(input_combinations_list_dict):
            result = self.train_clf(input_combination, inputs, labels, i)
            results.append(result)

        self.results = results
        return results

    def get_best_params(self):
        """
        In results search for min acc. Return parameters for that model.
        :return:
        """
        return min(self.results, key=lambda x: x['score']['acc'])

    def save_report(self):
        """
        Returns report as csv.
        Header are model params with acc and rsme.
        """
        # flat results
        report_data = []
        for result in self.results:
            flat = result['params'].copy()
            flat.update(result['score'])
            report_data.append(flat)

        df = pd.DataFrame(report_data)
        df.to_csv('results/grid_search_results.csv', index=False, na_rep='None')

