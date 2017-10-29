import numpy as np
import pandas as pd
import xarray as xr

from sklearn.base import BaseEstimator,TransformerMixin

from .utils import create_interpolator, events_to_xr_dim
from .spike import Smoother, DEFAULT_TAU

class PeriEventTraceSampler(BaseEstimator,TransformerMixin):
    """docstring for EventTensorizer."""
    def __init__(self, traces, sample_times):
        self.sample_times = sample_times
        self.traces = traces

    def fit(self, X, y=None):
        self.splined_traces_ = self.traces.apply(
            lambda y: create_interpolator(self.traces.index,y),
            axis=0,
        )
        return self

    def transform(self, X):

        # define a local function that will extract traces around each event
        def extractor(ev):
            t = self.sample_times + ev['time']
            interpolated = self.splined_traces_.apply(
                lambda s: pd.Series(s(t),index=self.sample_times)
                )
            return xr.DataArray(interpolated.T,dims=['sample_times','neuron'])

        # do the extraction
        tensor = [extractor(ev) for _,ev in X.iterrows()]
        concat_dim = events_to_xr_dim(X)

        # concatenate the DataArrays into a single DataArray
        return xr.concat(tensor,dim=concat_dim)


class PeriEventSpikeSampler(BaseEstimator,TransformerMixin):
    """docstring for PeriEventSpikeSampler."""
    def __init__(self, spikes, sample_times, tracizer=None,tracizer_kwargs=None):
        self.spikes = spikes
        self.sample_times = sample_times
        self.Tracizer = tracizer
        self.tracizer_kwargs = tracizer_kwargs

    def fit(self, X, y=None):
        if self.Tracizer is None:
            self.Tracizer = Smoother
        if self.tracizer_kwargs is None:
            self.tracizer_kwargs = dict()

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

            traces.index = self.sample_times[:-1]

            return xr.DataArray(traces,dims=['sample_times','neuron'])

        # do the extraction
        tensor = [extractor(ev) for _,ev in X.iterrows()]
        concat_dim = events_to_xr_dim(X)

        # concatenate the DataArrays into a single DataArray
        return xr.concat(tensor,dim=concat_dim)
