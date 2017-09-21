import numpy as np
import xarray as xr
from sklearn.base import TransformerMixin

class BaseTensorizer(TransformerMixin):
    """docstring for SpikeTensorizer."""
    def __init__(self, events, bins, range=None):
        super(BaseTensorizer, self).__init__()
        self.events = events
        _, bins = np.histogram([],bins,range)
        self.bins = bins

        # builds the event dataframe into coords for xarray
        coords = (
            self.events
            .to_dict(orient='list')
            )
        coords = {k:('event',v) for k,v in coords.items()}

        # define a DataArray that will describe the event dimension
        self.concat_dim = xr.DataArray(
            self.events.index,
            name='event',
            dims=['event'],
            coords=coords,
            )

        # TODO: use property decorator to keep self.events and self.concat_dim in sync
        
