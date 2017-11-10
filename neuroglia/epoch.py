import numpy as np
import pandas as pd
import xarray as xr

from sklearn.base import BaseEstimator,TransformerMixin

class EpochReducer(BaseEstimator,TransformerMixin):
    """docstring for EpochReducer."""
    def __init__(self, traces, method='mean'):
        self.traces = traces
        self.method = method

    def fit(self, X, y=None):
        if self.method == 'mean':
            self.method = np.mean
        elif self.method == 'max':
            self.method = np.max

        return self

    def transform(self, X):

        # define a local function that will extract traces around each event
        def extractor(ev):
            window = ev['time'], ev['time'] + ev['duration']
            mask = (
                (X.index >= ev['time'])
                & (X.index < (ev['time'] + ev['duration']))
                )
            return X[mask].apply(self.method,axis=0)

        # do the extraction
        tensor = [extractor(ev) for _,ev in X.iterrows()]
        concat_dim = events_to_xr_dim(X)

        # concatenate the DataArrays into a single DataArray
        return xr.concat(tensor,dim=concat_dim)
