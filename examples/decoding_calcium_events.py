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

###############################################
# Output::
#               541510267  541510270  541510307  541510405  588381938  541510410    ...      541509981  541509952  541510950  541511172  541509957  541511118
#     10.30338   0.219740   0.151908   0.188632   0.668425   0.081433   0.183986    ...       0.226459   0.065945   0.079813   0.084233   0.360267   0.156850
#     10.33655   0.167939   0.142997   0.178289   0.479256   0.089889   0.073484    ...       0.152407   0.127615   0.151770   0.090351   0.308192   0.108658
#     10.36972   0.136697   0.068048   0.163631   0.533209   0.023668   0.135786    ...       0.059235   0.070546   0.114366   0.106841   0.295819   0.033781
#     10.40289   0.157216   0.105795   0.089224   0.456814   0.004756   0.054532    ...       0.138155   0.025575   0.073771   0.085469   0.329818   0.098048
#     10.43606   0.130490   0.097038   0.099539   0.381433   0.105907   0.050581    ...       0.116505   0.037647   0.092018   0.100030   0.487283   0.066526
#
#     [5 rows x 154 columns]


#############################################
# Next, we'll load stim_table

stim_table = nwb_dataset.get_stimulus_table('natural_scenes')
print(stim_table.head())

#############################################
# Output::
#        frame  start    end
#     0     92  16126  16133
#     1     27  16134  16141
#     2     52  16141  16148
#     3     37  16149  16156
#     4    103  16156  16163

#############################################
# The stim_table lists stimulus times in terms of the start and end frames of
# the calcium traces, but we need start times and durations for neuroglia, so
# we'll need to reshape

stim_table['time'] = timestamps[stim_table['start']]
stim_table['duration'] = timestamps[stim_table['end']+1] - stim_table['time']
stim_table.rename(columns={'frame':'image_id'},inplace=True)

print(stim_table.head())

##############################################
# Output::
#        image_id  start    end       time  duration
#     0        92  16126  16133  545.22975   0.26537
#     1        27  16134  16141  545.49512   0.26538
#     2        52  16141  16148  545.72733   0.26537
#     3        37  16149  16156  545.99270   0.26538
#     4       103  16156  16163  546.22491   0.26538



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

##############################################
# Output::
#     Calcium Traces Score: 0.05 (chance: 0.01)

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


##############################################
# Output::
#     Calcium Traces Score: 0.18 (chance: 0.01)
