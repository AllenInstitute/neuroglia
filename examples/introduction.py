#!/usr/bin/env python
"""
Introduction to neuroglia
============

Neuroglia is a Python library for analyzing large scale electrophysiology and calcium imaging with scikit-learn machine learning pipelines

Applying modern machine learning techniques to the analysis of neurophysiology data requires the researcher to extract relevant features from the continuous time-varying activity of populations of recorded neurons. For example, to apply supervised classification techniques to population activity for a decoding analysis, the researcher must create a population response vector for each stimulus to decode. Depending on the scientific question and recording modality, this response vector could be the mean calcium signal in a window during each stimulus or the time to first spike after the stimulus onset.

In neuroglia, these transformations between these core data structures are defined as scikit-learn compatible transformers—Python objects utilizing a standardized fit, transform, and predict methods, allow them to be chained together into scikit-learn pipelines.

Neuroglia helps you transform your data between the three canonical shapes of neurophysiology data:

-	**Spikes**: a list of timestamps labelled with the neuron that elicited the spike
-	**Traces**: a 2D array of neurons x time. For example, calcium traces from 2P, binned spikin activity, or an LFP signal.
-	**Tensor**: a 3D array of traces aligned to events (events x neurons x time)

Transformations between these representations are implemented as scikit-learn transformers. This means that they all are defined as objects with “fit” and “transform” methods so that, for example, applying a Gaussian smoothing to a population of spiking data means transforming from a “spike times” structure to a “traces” structure like so:

In this example, we are going to demonstrate neuroglia's use by implementing a decoding analysis with spikes recorded from a Neuropixels probe in mouse primary visual cortex.

"""

"""####################################
# first, we need to load the data



import pandas as pd
ephys_data = pd.read_pickle('/allen/aibs/mat/RamIyer/frm_Dan/NWBFilesSev/V1_NI_pkl_data/M15_ni_data.pkl')
spikes = ephys_data['spike_times']
events = ephys_data['stim_table'].rename(
    columns={
        'Start':'time',
        'Frame':'image_id',
    },
)

from neuroglia.nwb import SpikeTablizer
spikes = SpikeTablizer().fit_transform(ephys_data['spiketimes'])
print(spikes.head())"""

####################################
# Spikes
# ------
#
# We start with `spikes`, a pandas dataframe containing spike times and neuron labels for each spike recorded and sorted.
#

print(spikes.head())

####################################
# Note: depending on how your data is stored, you may need to reshape it to match this structure. If you are loading spike times from an NWB file, the SpikeTablizer in the neuroglia.nwb module may be useful.
#
# Events
# ------
#
# Events are events that we wish to align neural data to. For example, stimulus presentations, behavioral events, or even other neural events.
#
# In this example, `events` is a series of visual stimulus presentations, including the time of the presentation and an identifier for each image.

print(events.head())

####################################
# PeriEventSpikeSampler
# ---------------------
#
# We want to extract the spikes that were elicited by each event, returning a 3D Tensor
#
# We do this using the PeriEventSpikeSampler. We initialize it by passing in the spikes we want to sample from.

import numpy as np
from neuroglia.event import PeriEventSpikeSampler

bins = np.arange(0.1, 0.35, 0.01)

spike_sampler = PeriEventSpikeSampler(
    spikes=spikes,
    sample_times=bins,
)


####################################
# Now we're ready to sample spikes for each event. We do so using the scikit-learn style `fit_transform()` syntax, passing `events` in as X. The PeriEventSpikeSampler returns a tensor with each event labelled with the timeseries of spiking activity relative to the event.

tensor = spike_sampler.fit_transform(events)
print(tensor)

####################################
# We can get the total number of elicited spikes count with the `ResponseReducer` transformer

from neuroglia.tensor import ResponseReducer
reducer = ResponseReducer(func=np.sum)


####################################
# Again, we use the scikit-learn style `fit_transform()` syntax to transform the tensor into a 2D array, where each row contains the population response of each event.

population_response = reducer.fit_transform(tensor)
print(population_response)

####################################
# Pipelines
# ---------
# Because neuroglia transformers are scikit-learn compatible, we can take advantage of scikit-learn features like Pipelines, chaining each transformer together into a single pipeline, implementing feature extraction, normalization, dimensionality reduction, and decoding.

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

pipeline = Pipeline([
    ('spike_sampler', PeriEventSpikeSampler(spikes=spikes, sample_times=bins)),
    ('extract', ResponseReducer(func=np.sum)),
    ('rescale', StandardScaler()),
    ('classify', LinearDiscriminantAnalysis()),
])

####################################
# We'll use scikit-learn's model_selection module to split our set of events into a training and testing set.

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    events, events['image_id'],test_size=0.33,
)


####################################
# Next, we'll train the entire pipeline.

pipeline.fit(X_train,y_train)

####################################
# Finally we'll test the pipeline's performance on the held out data

score = pipeline.score(X_test,y_test)

n_images = len(events['image_id'].unique())
print(score*n_images)
