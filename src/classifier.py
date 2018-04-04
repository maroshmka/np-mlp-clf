import random

import numpy as np

from src import func
from src.mlp import MLP
from src.util import onehot_decode, onehot_encode


class MLPClassifier(MLP):
    def __init__(self, dim_in, n_classes, *args, **kwargs):
        self.n_classes = n_classes
        super().__init__(dim_in=dim_in, dim_out=n_classes, *args, **kwargs)

    def set_params(self, **parameters):
        for parameter, value in parameters.items():
            setattr(self, parameter, value)
        return self

    def cost(self, targets, outputs, count):
        return func.MSE(targets, outputs)        

    def predict(self, inputs):
        # outputs, *_ = self.forward(inputs)  # if self.forward() can take a whole batch
        outputs = np.stack([self.forward(x)[0] for x in inputs.T])
        return onehot_decode(outputs, axis=1)

    def test(self, inputs, labels):
        dim, count = inputs.shape
        outputs = self.predict(inputs)
        targets = onehot_encode(labels, self.n_classes)
        predicted = onehot_encode(outputs, self.n_classes)

        missed = labels != outputs
        CE = np.sum(missed / count)
        RE = np.sum(self.cost(targets, predicted, count)) / count
        return CE, RE

    def train(self, inputs, labels, verbosity=0):
        (_, count) = inputs.shape
        targets = onehot_encode(labels, self.n_classes)
        training_data = list(zip(inputs.T, targets.T))

        CEs = []
        REs = []

        for ep in range(self.eps):
            if verbosity > 1:
                print('Ep {:3d}/{}: '.format(ep + 1, self.eps), end='')

            CE = 0
            RE = 0

            random.shuffle(training_data)
            mini_batches = [
                training_data[k:k + self.mini_batch_size]
                for k in range(0, len(training_data), self.mini_batch_size)
            ]

            for mini_batch in mini_batches:
                dW_hid = None
                dW_out = None

                for x, d in mini_batch:
                    y, dW_hid, dW_out = self.backward(x, d, count)

                    CE += onehot_decode(d, axis=0) != onehot_decode(y, axis=0)
                    RE += self.cost(d, y, count)

                    dW_hid += dW_hid
                    dW_out += dW_out

                self.W_hid += (self.alpha/self.mini_batch_size) * dW_hid
                self.W_out += (self.alpha/self.mini_batch_size) * dW_out

            CE /= count
            RE /= count

            CEs.append(CE)
            REs.append(RE)

            if verbosity > 1:
                print('CE = {:6.2%}, RE = {:.5f}'.format(CE, RE))

        if verbosity > 0:
            print()

        return CEs, REs
