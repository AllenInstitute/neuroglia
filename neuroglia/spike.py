import numpy as np
import pandas as pd
from sklearn.base import TransformerMixin
from neuroglia.core import BaseTensorizer


class Binarizer(TransformerMixin):
    """docstring for Binarizer."""
    def __init__(self,binsize,neuron_col='cluster',time_col='time'):
        super(Binarizer, self).__init__()
        self.binsize = binsize
        self.neuron_col = neuron_col
        self.time_col = time_col

    def fit(self, X, y=None):
        self.time_bins = np.arange(
            X[self.time_col].min(),
            X[self.time_col].max(),
            self.binsize,
            )
        self.neurons = X[self.neuron_col].unique()
        return self

    def transform(self, X):

        traces = []
        for neuron in neurons:
            neuron_mask = X[self.neuron_col]==neuron
            hist, bins = np.histogram(
                X[neuron_mask][self.time_col],
                self.time_bins
            )
            traces.append(
                pd.Series(data=hist,index=bins[:-1],name=neuron)
                )
        return pd.concat(traces,axis=1)

class Convolver(TransformerMixin):
    """docstring for Convolver."""
    def __init__(self, arg):
        super(Convolver, self).__init__()
        self.arg = arg

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        raise NotImplementedError


class SpikeTensorizer(BaseTensorizer):
    """docstring for SpikeTensorizer."""
    def __init__(self, events):
        super(SpikeTensorizer, self).__init__(events)

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        raise NotImplementedError
