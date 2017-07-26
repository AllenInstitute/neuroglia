from sklearn.base import TransformerMixin

class Raveller(TransformerMixin):
    """docstring for Raveller."""
    def __init__(self, arg):
        super(Raveller, self).__init__()
        self.arg = arg
