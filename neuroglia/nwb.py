import pandas as pd
from sklearn.base import BaseEstimator,TransformerMixin

class SpikeTablizer(BaseEstimator,TransformerMixin):
    """Convert a dictionary of spike times to a dataframe of spike times.

    It is common to store spike times as a dictionary where the keys are neuron
    IDs and the values are arrays of spike times for a given neuron.

    This transformer converts a dictionary of spike times into a table of spike
    times.

    Examples
    --------

    >>> import numpy as np
    >>> import pandas as pd
    >>> from neuroglia.nwb import SpikeTablizer
    >>> binner = SpikeTablizer()
    >>> spike_dict = {0:[0.1,0.2,0.3],2:[0.11]}
    >>> spikes = binner.fit_transform(spike_dict)

    See also
    --------

    neuroglia.spike.Smoother
    neuroglia.spike.Binner

    Notes
    -----

    This estimator is stateless (besides constructor parameters), the
    fit method does nothing but is useful when used in a pipeline.

    """
    def __init__(self):
        pass

    def fit(self, X, y=None):       # pragma: no cover
        """ Do nothing an return the estimator unchanged.

        This method is here to implement the scikit-learn API and work in
        scikit-learn pipelines.

        Parameters
        ----------

        X : dictionary of spike times in the format {<neuron>:<spike_times}
        y : (ignored)

        Returns
        -------

        self
        """
        return self

    def transform(self, X):
        """ Convert a dictionary of spike times to a dataframe of spike times.

        Parameters
        ----------
        X : dictionary of spike times in the format {<neuron>:<spike_times}

        Returns
        -------
        Xt : pandas DataFrame with columns ['time','neuron']
        """
        population = {'neuron':[],'time':[]}
        for n,times in X.items():
            for t in times:
                population['neuron'].append(n)
                population['time'].append(t)
        df = pd.DataFrame(population).sort_values('time')
        df.reset_index(drop=True, inplace=True)
        return df
