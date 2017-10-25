import numpy as np
import pandas as pd
import xarray as xr
from scipy import stats
from sklearn.base import TransformerMixin, BaseEstimator


def get_neuron(neuron_spikes):
    unique_neurons = neuron_spikes['neuron'].unique()
    assert len(unique_neurons)==1
    return unique_neurons[0]

class Binarizer(BaseEstimator,TransformerMixin):
    """Binarize a population of spike events into an array of spike counts.

    """
    def __init__(self,sample_times):
        self.sample_times = sample_times

    def fit(self, X, y=None):
        return self

    def __make_trace(self,neuron_spikes):
        neuron = get_neuron(neuron_spikes)

        trace, _ = np.histogram(
            neuron_spikes['time'],
            self.sample_times
            )
        return pd.Series(data=trace,index=self.sample_times[:-1],name=neuron)

    def transform(self, X):
        traces = X.groupby('neuron').apply(self.__make_trace).T
        return traces

KERNELS = {
    'gaussian': stats.norm,
    'exponential': stats.expon,
    'boxcar': stats.uniform,
}

DEFAULT_TAU = 0.005

class Smoother(BaseEstimator,TransformerMixin):
    """docstring for Smoother."""
    def __init__(self,sample_times,kernel='gaussian',tau=DEFAULT_TAU):

        self.sample_times = sample_times

        self.kernel = kernel
        self.tau = tau

    def fit(self, X, y=None):
        return self

    def __make_trace(self,neuron_spikes):
        neuron = get_neuron(neuron_spikes)

        kernel_func = lambda spike: KERNELS[self.kernel](loc=spike,scale=self.tau)

        data = (
            neuron_spikes['time']
            .map(lambda t: kernel_func(t).pdf(self.sample_times)) # creates kernel function for each spike and applies to the time bins
            .sum() # and adds them together
        )

        return pd.Series(
            data=data,
            index=self.sample_times,
            name=neuron,
            )

    def transform(self, X):
        traces = X.groupby('neuron').apply(self.__make_trace).T
        if len(traces)==0:
            traces = pd.DataFrame(index=self.sample_times)
        return traces
