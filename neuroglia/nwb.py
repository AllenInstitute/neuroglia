import pandas as pd
from sklearn.base import TransformerMixin

class SpikeTablizer(TransformerMixin):
    """Binarize a population of spike events into an array of spike counts.

    """
    def __init__(self):
        super(SpikeTablizer, self).__init__()

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        population = {'neuron':[],'time':[]}
        for n,times in X.items():
            for t in times:
                population['neuron'].append(n)
                population['time'].append(t)
        return pd.DataFrame(population).sort_values('time')
