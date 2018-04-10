import numpy as np
import pandas as pd
import xarray as xr

from sklearn.base import BaseEstimator,TransformerMixin

from .utils import create_interpolator, events_to_xr_dim
from .spike import Binner, DEFAULT_TAU

class PeriEventTraceSampler(BaseEstimator,TransformerMixin):
    """Take event-aligned samples of traces from a population of neurons.

    Traces are sampled relative to the event time. There is no enforced
    constraint that the times of events or sample_times relative to the events
    need to align to trace sample times. Rather, samples are interpolated from
    the values in the traces DataFrame.

    Parameters
    ----------
    traces : pandas DataFrame with 'time' as the index and neuron IDs in columns
        The traces that will be sampled from when the transform method is called
    sample_times : array
        Time relative to events that will be used to sample or bin spikes.

    Notes
    -----

    This estimator is stateless (besides constructor parameters), the
    fit method does nothing but is useful when used in a pipeline.
    """
    def __init__(self, traces, sample_times):
        self.sample_times = sample_times
        self.traces = traces

    def _make_splined_traces(self):
        self.splined_traces_ = self.traces.apply(
            lambda y: create_interpolator(self.traces.index,y),
            axis=0,
        )

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
        """Sample traces around each event

        Parameters
        ----------
        X : pandas.DataFrame with a column named 'time'

        Returns
        -------
        Xt : xarray.DataArray with columns ['event','sample_time','neuron']
        """
        self._make_splined_traces()

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

class PeriEventTraceReducer(BaseEstimator,TransformerMixin):
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
    def __init__(self, traces, sample_times):
        self.traces = traces
        self.sample_times = sample_times

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

        sample_dim = xr.DataArray(
            self.sample_times[:-1],
            name='sample_time',
            dims=['sample_time'],
        )

        # define a local function that will extract traces around each event
        def extractor(ev):
            bins = self.sample_times + ev['time']

            local_extract = []
            for window in zip(bins[:-1],bins[1:]):
                mask = (
                    (self.traces.index >= window[0])
                    & (self.traces.index < window[1])
                    )
                local_extract.append(
                    (
                        self.traces[mask]
                        .apply(self.func,axis=0)
                        .to_xarray()
                        .rename({'index':'neuron'})
                    )
                )
            return xr.concat(local_extract,dim=sample_dim)

        # do the extraction
        tensor = [extractor(ev) for _,ev in X.iterrows()]
        concat_dim = events_to_xr_dim(X)

        # concatenate the DataArrays into a single DataArray
        return xr.concat(tensor,dim=concat_dim)


class PeriEventSpikeSampler(BaseEstimator,TransformerMixin):
    """Take event-aligned samples of spikes from a population of neurons.

    Parameters
    ----------
    spikes : pandas DataFrame with columns ['time','neurons']
        The spikes that will be sampled from when the transform method is called
    sample_times : array
        Time relative to events that will be used to sample or bin spikes.
    fillna : boolean, optional (default: True)
        Whether to fill unobserved values. This is likely to occur if a given
        event has no spikes associated with it.
    sampler : transformer, optional (default: neuroglia.spikes.Binner)
        Binner or Smoother from neuroglia.spikes
    sampler_kwargs : dict-like
        Dictionary of keyword arguments to pass along to the Sampler

    Notes
    -----

    This estimator is stateless (besides constructor parameters), the
    fit method does nothing but is useful when used in a pipeline.
    """
    def __init__(self, spikes, sample_times, fillna=True, sampler=None,sampler_kwargs=None):
        self.spikes = spikes
        self.sample_times = sample_times
        self.fillna = fillna
        self.Sampler = sampler
        self.sampler_kwargs = sampler_kwargs

    def _assign_sampler(self):
        if self.Sampler is None:
            self.Sampler = Binner
        if self.sampler_kwargs is None:
            self.sampler_kwargs = dict()

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
        """Sample spikes around each event

        Parameters
        ----------
        X : pandas.DataFrame with a column named 'time'

        Returns
        -------
        Xt : xarray.DataArray with columns ['event','sample_time','neuron']
        """

        self._assign_sampler()

        # define a local function that will extract traces around each event
        def extractor(ev):

            t = self.sample_times + ev['time']
            start = t[0] - 10*DEFAULT_TAU
            stop = t[-1] + 10*DEFAULT_TAU

            local_mask = (
                (self.spikes['time']>start) & (self.spikes['time']<stop) # TODO: replace with np.search_sorted to speed up this query
            )
            local_spikes = self.spikes[local_mask]

            sampler = self.Sampler(t,**self.sampler_kwargs)
            traces = sampler.fit_transform(local_spikes)

            traces.index = self.sample_times[:len(traces)]

            return xr.DataArray(traces,dims=['sample_times','neuron'])

        # do the extraction
        tensor = [extractor(ev) for _,ev in X.iterrows()]
        concat_dim = events_to_xr_dim(X)

        # concatenate the DataArrays into a single DataArray
        tensor = xr.concat(tensor,dim=concat_dim)

        if self.fillna:
            tensor = tensor.fillna(0)

        return tensor
