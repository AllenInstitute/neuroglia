
from sklearn.base import TransformerMixin

class BaseTensorizer(TransformerMixin):
    """docstring for SpikeTensorizer."""
    def __init__(self, events):
        super(BaseTensorizer, self).__init__()
        self.events = events
