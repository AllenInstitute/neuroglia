import numpy as np
import xarray as xr
from sklearn.base import TransformerMixin

def events_to_xr_dim(events,dim='event'):
    # builds the event dataframe into coords for xarray
    coords = events.to_dict(orient='list')
    coords = {k:(dim,v) for k,v in coords.items()}
    # define a DataArray that will describe the event dimension
    return xr.DataArray(
        events.index,
        name=dim,
        dims=[dim],
        coords=coords,
    )

class BaseTensorizer(TransformerMixin):
    """docstring for SpikeTensorizer."""
    def __init__(self, events, bins, range):
        super(BaseTensorizer, self).__init__()
        self.events = events
        _, bins = np.histogram([],bins,range)
        self.bins = bins
        self.concat_dim = events_to_xr_dim(self.events)

        # TODO: use property decorator to keep self.events and self.concat_dim in sync
        