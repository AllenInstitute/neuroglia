#!/usr/bin/env python
"""
natural scene decoding ophys
==============================

This is an example of how to decode natural images from calcium imaging traces in V1

"""


#######################################
# First, let's download an experiment from the Allen Institute Brain Observatory
from allensdk.core.brain_observatory_cache import BrainObservatoryCache

boc = BrainObservatoryCache()

OEID = 541206592
nwb_dataset = boc.get_ophys_experiment_data(OEID)

#############################################
# Next, we'll load the dF/F traces and put them in a DataFrame


timestamps, dff = nwb_dataset.get_dff_traces()
neuron_ids = nwb_dataset.get_cell_specimen_ids()

import pandas as pd
traces = pd.DataFrame(
    dff.T,
    columns=neuron_ids,
    index=timestamps,
)

print(traces.head())


#############################################
# Next, we'll load stim_table

stim_table = nwb_dataset.get_stimulus_table('natural_scenes')
print(stim_table.head())

#############################################
# The stim_table lists stimulus times in terms of the start and end frames of
# the calcium traces, but we need start times and durations for neuroglia, so
# we'll need to reshape

stim_table['time'] = timestamps[stim_table['start']]
stim_table['duration'] = timestamps[stim_table['end']+1] - stim_table['time']
stim_table.rename(columns={'frame':'image_id'},inplace=True)

print(stim_table.head())

###########################################
# Reduce the traces to responses

import numpy as np
from neuroglia.epoch import EpochTraceReducer

reducer = EpochTraceReducer(traces,func=np.mean)
X = reducer.fit_transform(stim_table[['time','duration']])
y = stim_table['image_id']

###########################################
# Do some dimensionality reduction on the responses

from sklearn.decomposition import PCA
pca = PCA()
X_reduced = pca.fit_transform(X)


###########################################
# Plot the first two Principal Components

import matplotlib.pyplot as plt
plt.scatter(X_reduced[:,0],X_reduced[:,1],c=y,cmap='hsv')

# plt.show()


############################################
# Creating training and test sets

from sklearn.model_selection import train_test_split

X = stim_table[['time','duration']]
y = stim_table['image_id']

chance = 1.0 / len(y.unique())

X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.33)


############################################
# Create a pipeline to decode image

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

calcium_traces_pipeline = Pipeline([
    ('reducer', EpochTraceReducer(traces,func=np.mean)),
    ('rescale', StandardScaler()),
    ('classify', LinearDiscriminantAnalysis()),
])

calcium_traces_pipeline.fit(X_train,y_train)
calcium_traces_score = calcium_traces_pipeline.score(X_test,y_test)

print("Calcium Traces Score: {:0.2f} (chance: {:0.3f})".format(calcium_traces_score,chance))


#############################################
# Extract calcium events

from neuroglia.calcium import CalciumDeconvolver

deconvolver = CalciumDeconvolver()
calcium_events = deconvolver.fit_transform(traces)

calcium_events_pipeline = Pipeline([
    ('reducer', EpochTraceReducer(calcium_events,func=np.mean)),
    ('rescale', StandardScaler()),
    ('classify', LinearDiscriminantAnalysis()),
])

calcium_events_pipeline.fit(X_train,y_train)
calcium_events_score = calcium_events_pipeline.score(X_test,y_test)

print("Calcium Events Score: {:0.2f} (chance: {:0.3f})".format(calcium_events_score,chance))
