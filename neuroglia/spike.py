import numpy as np
import pandas as pd
import xarray as xr
from scipy import stats
from sklearn.base import TransformerMixin

from .utils import create_bin_array


def get_neuron(neuron_spikes):
    unique_neurons = neuron_spikes['neuron'].unique()
    assert len(unique_neurons)==1
    return unique_neurons[0]

class Binarizer(TransformerMixin):
    """Binarize a population of spike events into an array of spike counts.



    """
    def __init__(self,bins,window=None,**hist_kwargs):
        super(Binarizer, self).__init__()

        try:
            window = hist.pop('range')
        except KeyError:
            pass

        # use bins & window params to create a time axis
        self.bins = create_bin_array(bins,window)
        self.t = self.bins[:-1]

        self.hist_kwargs = hist_kwargs.copy()

    def fit(self, X, y=None):
        return self

    def __make_trace(self,neuron_spikes):
        neuron = get_neuron(neuron_spikes)

        trace, _ = np.histogram(
            neuron_spikes['time'],
            self.bins,
            **self.hist_kwargs
            )
        return pd.Series(data=trace,index=self.t,name=neuron)

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
    def __init__(self,bins,window=None,kernel='gaussian',tau=DEFAULT_TAU):
        super(Smoother, self).__init__()
        # self.bins = bins
        # self.window = window
        self.kernel = kernel
        self.tau = tau

        # use bins & window params to create a time axis
        self.bins = create_bin_array(bins,window)
        self.t = self.bins[:-1]

    def fit(self, X, y=None):
        return self

    def __make_trace(self,neuron_spikes):
        neuron = get_neuron(neuron_spikes)

        kernel_func = lambda spike: KERNELS[self.kernel](loc=spike,scale=self.tau)

        data = (
            neuron_spikes['time']
            .map(lambda t: kernel_func(t).pdf(self.bins)) # creates kernel function for each spike and applies to the time bins
            .sum() # and adds them together
        )

        # normalize the data
        data = np.multiply(data[:-1],np.diff(self.bins))

        return pd.Series(
            data=data,
            index=self.t,
            name=neuron,
            )

    def transform(self, X):
        traces = X.groupby('neuron').apply(self.__make_trace).T
        if len(traces)==0:
            traces = pd.DataFrame(index=self.t)
        return traces
