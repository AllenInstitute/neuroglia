# neuroglia

pipelines for transformations for neurophysiology data

## Usage

These are just pseudocode to demonstrate how the package might work.

### make a PSTH

```
spikes = Spikes() # spikes
events = Events()

# BINS is a transformer that transforms spikes into a histogram
psth = SpikeTensorizer(
    events=events, # all Tensorizers need events (either a dataframe or list of dicts or list of times)
    method='histogram',
    bins=30,
    range=(-1,2.0),
    )

tensor = psth.fit_transform(spikes) # spikes to tensor


```

### tensorize a calcium signal

```
events = Events()
calcium = CalciumTraces() # matrix / continuous

trace_tensorizer = TraceTensorizer(
    events=events,
    range=(-1.0,2.0),
    )
tensor = trace_tensorizer.transform(calcium)

# do tensor decomposition
cptd = TensorDecomposition(rank=5)
cptd.fit(tensor)
factors = cptd.transform(tensor)

# visualize factors

reconstruction = cptd.inverse_transform(factors[1:3])
```

### Synthesize a calcium signal

```

# CalciumVectorizer is a transformer that transforms spikes into a calcium signal
calcium_vectorizer = CalciumVectorizer(indicator='GCamP6s')

calcium_traces = calcium_vectorizer.fit_transform(spikes)

# do PCA from scikit-learn
```


### Infer spikes from a calcium signal

```
calcium = Calcium()
spike_finder = SpikeFinder()

spike_finder.fit(calcium) # continuous to spikes

found_spikes = spike_finder.transform(calcium)

```

## Installation

`pip install neuroglia`

### Requirements

## License

## Authors

`neuroglia` was written by `Justin Kiggins <justink@alleninstitute.org>`.
