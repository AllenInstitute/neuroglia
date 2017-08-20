import numpy as np
from scipy import interpolate
import pandas as pd
import xarray as xr
from neuroglia.core import BaseTensorizer


class TraceTensorizer(BaseTensorizer):
    """docstring for SpikeTensorizer."""
    def __init__(self, events, **hist_kwargs):
        super(TraceTensorizer, self).__init__(events,**hist_kwargs)

    def fit(self, X, y=None):
        self.target_shape = len(self.events), len(self.bins), len(X.columns),
        dtypes = X.dtypes.unique()
        assert len(dtypes)==1
        self.target_dtype = dtypes[0]

        self.splines = X.apply(
            lambda y: interpolate.InterpolatedUnivariateSpline(X.index, y),
            axis=0,
        )
        return self

    def transform(self, X):

        # define a local function that will extract traces around each event
        def extractor(ev):
            bins = self.bins + ev['time']
            interpolated = self.splines.apply(
                lambda s: pd.Series(s(bins),index=self.bins)
                )
            return xr.DataArray(interpolated.T,dims=['time_from_event','neuron'])
        
        # do the extraction
        tensor = [extractor(ev) for _,ev in self.events.iterrows()]
        
        # builds the event dataframe into coords for xarray
        coords = (
            self.events
            .reset_index()
            .to_dict(orient='list')
            )
        coords = {k:('event',v) for k,v in coords.items()}

        # define a DataArray that will describe the event dimension
        concat_dim = xr.DataArray(
            self.events.index,
            name='event',
            dims=['event'],
            coords=coords,
            )
        
        # concatenate the DataArrays into a single DataArray
        return xr.concat(tensor,dim=concat_dim)
        