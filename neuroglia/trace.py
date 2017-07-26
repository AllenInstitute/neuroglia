from neuroglia.core import BaseTensorizer

class TraceTensorizer(BaseTensorizer):
    """docstring for SpikeTensorizer."""
    def __init__(self, events):
        super(TraceTensorizer, self).__init__(events)

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        raise NotImplementedError
