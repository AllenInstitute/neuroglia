import numpy as np
import pandas as pd
import xarray as xr
from scipy import stats
from sklearn.base import TransformerMixin, BaseEstimator


def get_neuron(neuron_spikes):
    unique_neurons = neuron_spikes['neuron'].unique()
    assert len(unique_neurons)==1
    return unique_neurons[0]

class Binner(BaseEstimator,TransformerMixin):
    """Bin a population of spike events into an array of spike counts.

    This transformer converts a table of spike times into a series of spike
    counts. Spikes are binned according to the spike_times argument.

    Parameters
    ----------

    sample_times : array-like
        The samples times that will be used to bin spikes.

    Attributes
    ----------


    Examples
    --------

    >>> import numpy as np
    >>> import pandas as pd
    >>> from neuroglia.spike import Binner
    >>> binner = Binner(np.arange(0,1.0,0.001))
    >>> spikes = pd.DataFrame({'times':np.random.rand})
    >>> X = binner.fit_transform(spikes)

    See also
    --------

    neuroglia.spike.Smoother
    neuroglia.nwb.SpikeTablizer

    """
    def __init__(self,sample_times):
        self.sample_times = sample_times

    def fit(self, X, y=None):
        """ Do nothing an return the estimator unchanged.

        This method is just there to implement the usual API and hence work in pipelines.

        Parameters
        ----------

        X : pandas DataFrame with columns ['time','neuron']
        y : (ignored)

        Returns
        -------

        self
        """
        return self

    def __make_trace(self,neuron_spikes):
        neuron = get_neuron(neuron_spikes)

        trace, _ = np.histogram(
            neuron_spikes['time'],
            self.sample_times
            )
        return pd.Series(data=trace,index=self.sample_times[:-1],name=neuron)

    def transform(self, X):
        """ Bin each neuron's spikes into a trace of spike counts.

        Parameters
        ----------
        X : pandas DataFrame with columns ['time','neuron']
            spike times that will be binned
        y : (ignored)

        Returns
        -------
        Xt : pandas DataFrame of spike counts
            Columns are neuron labels and the index is the left edge of the
            sample times.
        """
        traces = X.groupby('neuron').apply(self.__make_trace).T
        return traces

KERNELS = {
    'gaussian': stats.norm,
    'exponential': stats.expon,
    'boxcar': stats.uniform,
}

DEFAULT_TAU = 0.005

class Smoother(BaseEstimator,TransformerMixin):
    """Smooth a population of spike events into an array.

    This transformer converts a table of spike times into a trace of smoothed
    spike values. Spikes are binned according to the spike_times argument.

    Parameters
    ----------

    sample_times : array-like
        The samples times that will be used to bin spikes.

    Attributes
    ----------


    Examples
    --------

    >>> import numpy as np
    >>> import pandas as pd
    >>> from neuroglia.spike import Binner
    >>> binner = Binner(np.arange(0,1.0,0.001))
    >>> spikes = pd.DataFrame({'times':np.random.rand})
    >>> X = binner.fit_transform(spikes)

    See also
    --------

    neuroglia.spike.Smoother
    neuroglia.nwb.SpikeTablizer

    """
    def __init__(self,sample_times,kernel='gaussian',tau=DEFAULT_TAU):

        self.sample_times = sample_times

        self.kernel = kernel
        self.tau = tau

    def fit(self, X, y=None):
        """ Do nothing an return the estimator unchanged.

        This method is just there to implement the usual API and hence work in pipelines.

        Parameters
        ----------

        X : pandas DataFrame with columns ['time','neuron']
        y : (ignored)

        Returns
        -------

        self
        """
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
