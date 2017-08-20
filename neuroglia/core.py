import numpy as np
from sklearn.base import TransformerMixin

class BaseTensorizer(TransformerMixin):
    """docstring for SpikeTensorizer."""
    def __init__(self, events, **hist_kwargs):
        super(BaseTensorizer, self).__init__()
        self.events = events
        _,bins = np.histogram([],**hist_kwargs)
        self.bins = bins
