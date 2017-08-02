
from sklearn.base import TransformerMixin

class BaseTensorizer(TransformerMixin):
    """docstring for SpikeTensorizer."""
    def __init__(self, events, window):
        super(BaseTensorizer, self).__init__()
        self.events = events
        assert window[0]<window[1], 'window[0] must be less than window[1]'
        self.window = window
