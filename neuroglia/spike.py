import numpy as np
import pandas as pd
from scipy import stats
from sklearn.base import TransformerMixin
from neuroglia.core import BaseTensorizer


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
            **hist_kwargs,
            )
        return pd.Series(data=trace,index=bins[:-1],name=neuron)

    def transform(self, X):
        traces = X.groupby('neuron').apply(self.__make_trace).T
        return traces

class Smoother(TransformerMixin):
    """docstring for Smoother."""
    def __init__(self,bins,range=None,kernel='gaussian',tau=0.005):
        super(Smoother, self).__init__()
        self.bins = bins
        self.range = range
        KERNELS = {
            'gaussian': stats.norm,
            'exponential': stats.expon,
            'boxcar': stats.uniform,
        }
        self.kernel_func = KERNELS[kernel]
        self.tau = tau

    def fit(self, X, y=None):
        _, self.bins = np.histogram(X['time'],self.bins,self.range)
        return self

    def __make_trace(self,neuron_spikes):
        neuron = get_neuron(neuron_spikes)

        data = (
            neuron_spikes['time']
            .map(lambda x: self.kernel_func(loc=x,scale=self.tau))
            .map(lambda rv: rv.pdf(self.bins))
            .sum()
        )

        return pd.Series(
            data=data,
            index=self.bins,
            name=neuron,
            )

    def transform(self, X):
        traces = X.groupby('neuron').apply(self.__make_trace).T
        return traces

def align_spikes(spikes,time):
    aligned = {}
    for k,v in spikes.items():
        try:
            aligned[k] = v - time
        except TypeError:
            aligned[k] = [vv-time for vv in v]
    return aligned

def calc_n_bins(window,binsize):
    return int(np.round((window[1]-window[0])/binsize))-1

class SpikeTensorizer(BaseTensorizer):
    """docstring for SpikeTensorizer."""
    def __init__(self, events, window, binsize, **binarizer_kwargs):
        super(SpikeTensorizer, self).__init__(events, window)
        self.binsize = binsize
        self.binarizer_kwargs = binarizer_kwargs

    def fit(self, X, y=None):
        nbins = calc_n_bins(self.window,self.binsize)
        self.neurons = X.keys()
        self.target_shape = len(self.events), nbins, len(self.neurons),
        self.target_dtype = np.float_
        return self

    def transform(self, X):
        tensor = np.empty(self.target_shape,self.target_dtype)
        for ii,(_,ev) in enumerate(self.events.iterrows()):
            min_time,max_time = [t + ev['time'] for t in self.window]
            binarizer = Binarizer(
                binsize=self.binsize,
                min_time=min_time,
                max_time=max_time,
                **self.binarizer_kwargs,
                )
            binned = binarizer.fit_transform(align_spikes(X,ev['time']))
            tensor[ii,:,:] = binned.values
        return tensor
