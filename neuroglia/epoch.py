import numpy as np
import pandas as pd
import xarray as xr

from sklearn.base import BaseEstimator,TransformerMixin

from utils import events_to_xr_dim

class EpochTraceReducer(BaseEstimator,TransformerMixin):
    """Take event-aligned samples of traces from a population of neurons.

    Traces are sampled relative to the event time. There is no enforced
    constraint that the times of events or sample_times relative to the events
    need to align to trace sample times. Rather, samples are interpolated from
    the values in the traces DataFrame.

    Parameters
    ----------
    traces : pandas DataFrame with 'time' as the index and neuron IDs in columns
        The traces that will be sampled from when the transform method is called
    func : function
        Function that will be applied to trace samples within epochs

    Notes
    -----

    This estimator is stateless (besides constructor parameters), the
    fit method does nothing but is useful when used in a pipeline.
    """
    def __init__(self, traces, func):
        self.traces = traces
        self.func = func

    def fit(self, X, y=None):
        """Do nothing and return the estimator unchanged

        This method is here to implement the scikit-learn API and work in
        scikit-learn pipelines.

        Parameters
        ----------
        X : array-like

        Returns
        -------
        self

        """

        return self

    def transform(self, X):
        """Reduce traces around each event

        Parameters
        ----------
        X : pandas.DataFrame with a columns ['time','duration']

        Returns
        -------
        Xt : xarray.DataArray with columns ['event','neuron']
        """

        # define a local function that will extract traces around each event
        def extractor(ev):
            window = ev['time'], ev['time'] + ev['duration']
            mask = (
                (self.traces.index >= ev['time'])
                & (self.traces.index < (ev['time'] + ev['duration']))
                )
            return self.traces[mask].apply(self.func,axis=0).to_xarray()

        # do the extraction
        tensor = [extractor(ev) for _,ev in X.iterrows()]
        concat_dim = events_to_xr_dim(X)

        # concatenate the DataArrays into a single DataArray
        return xr.concat(tensor,dim=concat_dim)
