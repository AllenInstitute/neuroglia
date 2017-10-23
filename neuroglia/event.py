import numpy as np
from sklearn.base import BaseEstimator,TransformerMixin

from .trace import TraceTensorizer
from .utils import create_interpolator


class EventTraceTensorizer(BaseEstimator,TransformerMixin):
    """docstring for EventTensorizer."""
    def __init__(self, traces, bins, range):
        super(EventSpikeTensorizer, self).__init__()
        self.traces = traces
        self.bins = bins
        self.range = range

        self.splined_traces = traces.apply(
            lambda y: create_interpolator(traces.index,y),
            axis=0,
        )

    def fit(self, X, y=None):
        return self

    def transform(self, X):

        # define a local function that will extract traces around each event
        def extractor(ev):
            bins = self.bins + ev['time']
            interpolated = self.splined_traces.apply(
                lambda s: pd.Series(s(bins),index=self.bins)
                )
            return xr.DataArray(interpolated.T,dims=['time','neuron'])

        # do the extraction
        tensor = [extractor(ev) for _,ev in X.iterrows()]
        concat_dim = events_to_xr_dim(X)

        # concatenate the DataArrays into a single DataArray
        return xr.concat(tensor,dim=concat_dim)

class MeanResponseExtractor(BaseEstimator,TransformerMixin):
    """docstring for Annotator."""
    def __init__(self, traces,bins,range):
        super(MeanResponseExtractor, self).__init__()
        self.traces = traces
        self.bins = bins
        self.range = range

    def fit(self, X, y=None):
        return self

    def transform(self, X):

        tensorizer = EventTraceTensorizer(
            traces=self.traces,
            bins=self.bins,
            range=self.range,
            )
        tensor = tensorizer.fit_transform(X)
        return tensor.mean(dim='time').data

class EventSpikeTensorizer(BaseEstimator,TransformerMixin):
    """docstring for EventSpikeTensorizer."""
    def __init__(self, spikes, bins, range):
        super(EventSpikeTensorizer, self).__init__()
        self.spikes = traces
        self.bins = bins
        self.range = range

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
            bins = self.bins + ev['time']

            start = bins[0] - 10*DEFAULT_TAU,
            stop = bins[-1] + 10*DEFAULT_TAU

            local_mask = (
                (self.spikes['time']>start) & (self.spikes['time']<stop) # TODO: replace with np.search_sorted to speed up this query
            )
            local_spikes = self.spikes[local_mask]

            tracizer = self.Tracizer(bins,**self.tracizer_kwargs)
            traces = tracizer.fit_transform(local_spikes)
            traces.index = self.bins[:-1]

            return xr.DataArray(traces,dims=['time','neuron'])

        # do the extraction
        tensor = [extractor(ev) for _,ev in X.iterrows()]
        concat_dim = events_to_xr_dim(X)

        # concatenate the DataArrays into a single DataArray
        return xr.concat(tensor,dim=concat_dim)
