import func
from util import *


# Multi-Layer Perceptron
# (abstract base class)

class MLP:
    def __init__(self, dim_in, dim_out, dim_hid=20, mini_batch_size=1, eps=100, alpha=0.1,
                 hid_function='logsig', out_function='softmax', cost_function='square',
                 verbosity=1):

        assert hid_function in func.hid_activations, 'hid_function - `{hf}` is not in {hfs}'.format(
            hf=hid_function, hfs=func.hid_activations)
        assert out_function in func.out_activations, 'out_fucntion - `{of}` is not in {ofs}'.format(
            of=out_function, ofs=func.out_activations)
        assert cost_function in ['square',
                                 'ce'], 'cost function - `{cf}` has to be `square` or `ce`'.format(
            cf=cost_function)

        self.eps = eps
        self.verbosity = verbosity
        self.dim_in = dim_in
        self.dim_hid = dim_hid
        self.dim_out = dim_out

        self.mini_batch_size = mini_batch_size
        self.alpha = alpha

        self.hid_function = hid_function
        self.out_function = out_function
        self.cost_function = cost_function

        self.init_weights()

    def init_weights(self):
        np.random.seed(1)
        _interval = -1 / (np.sqrt(self.dim_in))
        self.W_hid = np.stack([np.random.uniform(-_interval, _interval, self.dim_in + 1) for x in
                               range(self.dim_hid)])
        self.W_out = np.stack([np.random.uniform(-_interval, _interval, self.dim_hid + 1) for x in
                               range(self.dim_out)])

    # activation functions & derivations
    # (not implemented, to be overriden in derived classes)

    def f_hid(self, x):
        return getattr(func, self.hid_function)(x)

    def df_hid(self, x):
        return getattr(func, 'd' + self.hid_function)(x)

    def f_out(self, x):
        return getattr(func, self.out_function)(x)

    def df_out(self, x):
        return getattr(func, 'd' + self.out_function)(x)

    # regularization

    def get_params(self):
        return {
            'eps': self.eps,
            'mini_batch_size': self.mini_batch_size,
            'dim_hid': self.dim_hid,
            'alpha': self.alpha,
            'hid_function': self.hid_function,
            'out_function': self.out_function,
        }

    def forward(self, x):
        a = self.W_hid @ augment(x)
        h = self.f_hid(a)

        b = self.W_out @ augment(h)
        y = self.f_out(b)

        return y, b, h, a

    # forward & backprop pass
    # (single input and target vector)

    def backward(self, x, d, count):
        y, b, h, a = self.forward(x)

        g_out = (d - y) * self.df_out(b)
        g_hid = (self.W_out[:, :-1].T @ g_out) * self.df_hid(a)

        dW_out = np.outer(g_out, augment(h)) 
        dW_hid = np.outer(g_hid, augment(x)) 

        return y, dW_hid, dW_out
