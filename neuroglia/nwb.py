import pandas as pd
from sklearn.base import TransformerMixin

class SpikeTablizer(TransformerMixin):
    """converts a dictionary of spike times in the form of neuron:[times] to a
    dataframe with "neuron" and "time" columns, sorted by "time"

    """
    def __init__(self):
        super(SpikeTablizer, self).__init__()

    def fit(self, X, y=None):       # pragma: no cover
        return self

    def transform(self, X):
        population = {'neuron':[],'time':[]}
        for n,times in X.items():
            for t in times:
                population['neuron'].append(n)
                population['time'].append(t)
        return pd.DataFrame(population).sort_values('time')
