import numpy as np
import pandas as pd
import xarray as xr

from sklearn.base import BaseEstimator,TransformerMixin

from .utils import create_interpolator, events_to_xr_dim, create_bin_array
from .spike import Smoother, DEFAULT_TAU

class EventTraceTensorizer(BaseEstimator,TransformerMixin):
    """docstring for EventTensorizer."""
    def __init__(self, traces, sample_times):
        self.sample_times = sample_times

        self.splined_traces = traces.apply(
            lambda y: create_interpolator(traces.index,y),
            axis=0,
        )

    def fit(self, X, y=None):
        return self

    def transform(self, X):

        # define a local function that will extract traces around each event
        def extractor(ev):
            t = self.sample_times + ev['time']
            interpolated = self.splined_traces.apply(
                lambda s: pd.Series(s(t),index=self.sample_times)
                )
            return xr.DataArray(interpolated.T,dims=['time_from_event','neuron'])

        # do the extraction
        tensor = [extractor(ev) for _,ev in X.iterrows()]
        concat_dim = events_to_xr_dim(X)

        # concatenate the DataArrays into a single DataArray
        return xr.concat(tensor,dim=concat_dim)


class EventSpikeTensorizer(BaseEstimator,TransformerMixin):
    """docstring for EventSpikeTensorizer."""
    def __init__(self, spikes, sample_times, tracizer=None,tracizer_kwargs=None):
        self.spikes = spikes
        self.sample_times = sample_times

        if tracizer_kwargs is None:
            tracizer_kwargs = dict()

        if tracizer is None:
            tracizer = Smoother

        self.Tracizer = tracizer
        self.tracizer_kwargs = tracizer_kwargs

    def fit(self, X, y=None):
        return self

    def transform(self, X):

        # define a local function that will extract traces around each event
        def extractor(ev):

            t = self.sample_times + ev['time']
            start = t[0] - 10*DEFAULT_TAU
            stop = t[-1] + 10*DEFAULT_TAU

            local_mask = (
                (self.spikes['time']>start) & (self.spikes['time']<stop) # TODO: replace with np.search_sorted to speed up this query
            )
            local_spikes = self.spikes[local_mask]

            tracizer = self.Tracizer(t,**self.tracizer_kwargs)
            traces = tracizer.fit_transform(local_spikes)

            traces.index = self.sample_times

            return xr.DataArray(traces,dims=['sample_times','neuron'])

        # do the extraction
        tensor = [extractor(ev) for _,ev in X.iterrows()]
        concat_dim = events_to_xr_dim(X)

        # concatenate the DataArrays into a single DataArray
        return xr.concat(tensor,dim=concat_dim)
