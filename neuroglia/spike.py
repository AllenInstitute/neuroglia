import numpy as np
import pandas as pd
import xarray as xr
from scipy import stats
from sklearn.base import TransformerMixin


def get_neuron(neuron_spikes):
    unique_neurons = neuron_spikes['neuron'].unique()
    assert len(unique_neurons)==1
    return unique_neurons[0]

class Binarizer(TransformerMixin):
    """Binarize a population of spike events into an array of spike counts.



    """
    def __init__(self,**hist_kwargs):
        super(Binarizer, self).__init__()
        # self.time_bins = np.arange(range[0],range[1],binsize)
        self.hist_kwargs = hist_kwargs

    def fit(self, X, y=None):
        hist,bins = np.histogram(X['time'],**self.hist_kwargs)
        self.time_bins = bins[:-1]

        self.neurons = X.keys()
        self.target_shape = len(self.neurons), len(self.time_bins)
        self.target_dtype = hist.dtype
        return self

    def __make_trace(self,neuron_spikes):
        neuron = get_neuron(neuron_spikes)

        hist_kwargs = self.hist_kwargs.copy()
        hist_kwargs.update(bins=self.time_bins)
        trace, bins = np.histogram(
            neuron_spikes['time'],
            **hist_kwargs
            )
        return pd.Series(data=trace,index=bins[:-1],name=neuron)

    def transform(self, X):
        traces = X.groupby('neuron').apply(self.__make_trace).T
        return traces

KERNELS = {
    'gaussian': stats.norm,
    'exponential': stats.expon,
    'boxcar': stats.uniform,
}

DEFAULT_TAU = 0.005

class Smoother(TransformerMixin):
    """docstring for Smoother."""
    def __init__(self,bins,range=None,kernel='gaussian',tau=DEFAULT_TAU):
        super(Smoother, self).__init__()
        self.bins = bins
        self.range = range
        self.kernel = kernel
        self.tau = tau

    def fit(self, X, y=None):
        _, self.bins = np.histogram(X['time'],self.bins,self.range)
        return self

    def __make_trace(self,neuron_spikes):
        neuron = get_neuron(neuron_spikes)

        kernel_func = lambda spike: KERNELS[self.kernel](loc=spike,scale=self.tau)

        data = (
            neuron_spikes['time']
            .map(lambda t: kernel_func(t).pdf(self.bins)) # creates kernel function for each spike and applies to the time bins
            .sum() # and adds them together
        )

        data = np.multiply(data[:-1],np.diff(self.bins))

        return pd.Series(
            data=data,
            index=self.bins[:-1],
            name=neuron,
            )

    def transform(self, X):
        traces = X.groupby('neuron').apply(self.__make_trace).T
        return traces
