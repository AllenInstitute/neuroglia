import numpy as np
from scipy import interpolate
import pandas as pd
import xarray as xr
from neuroglia.core import BaseTensorizer


class TraceTensorizer(BaseTensorizer):
    """docstring for SpikeTensorizer."""
    def __init__(self, events, bins, range=None):
        super(TraceTensorizer, self).__init__(events,bins,range)

    def fit(self, X, y=None):
        return self

    def transform(self, X):

        def create_interpolator(y):
            interpolator = interpolate.InterpolatedUnivariateSpline(X.index, y)
            return interpolator

        self.splines = X.apply(
            create_interpolator,
            axis=0,
        )
        # define a local function that will extract traces around each event
        def extractor(ev):
            bins = self.bins + ev['time']
            interpolated = self.splines.apply(
                lambda s: pd.Series(s(bins),index=self.bins)
                )
            return xr.DataArray(interpolated.T,dims=['time_from_event','neuron'])

        # do the extraction
        tensor = [extractor(ev) for _,ev in self.events.iterrows()]


        # concatenate the DataArrays into a single DataArray
        return xr.concat(tensor,dim=self.concat_dim)
