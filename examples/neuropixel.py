#!/usr/bin/env python
"""
natural scene decoding from V1
==============================

This is an example of how to decode natural images from spikes recorded in V1

"""

from __future__ import print_function

####################################
# first, we need to load the data

data_path = '/allen/aibs/mat/RamIyer/frm_Dan/NWBFilesSev/V1_NI_pkl_data/'

import pandas as pd
ephys_data = pd.read_pickle(data_path+'M15_ni_data.pkl')

####################################
# Let's get the dataframe of image presentations and rename the columns

events = ephys_data['stim_table'].rename(
    columns={
        'Start':'time',
        'Frame':'image_id',
    },
)
print(events.head())

####################################
# Next, let's reformat the spike times into a single table


from neuroglia.nwb import SpikeTablizer
spikes = SpikeTablizer().fit_transform(ephys_data['spiketimes'])
print(spikes.head())

####################################
# Now, we'll sample spikes near each event & build this into a xarray 3D tensor

from neuroglia.event import PeriEventSpikeSampler
import numpy as np
spike_sampler = PeriEventSpikeSampler(
    spikes=spikes,
    sample_times=np.arange(0.1,0.35,0.01),
)
tensor = spike_sampler.fit_transform(events)
print(tensor)

####################################
# We can get the average elicited spike count with the `ResponseReducer`

from neuroglia.tensor import ResponseReducer
import numpy as np
reducer = ResponseReducer(func=np.mean)
means = reducer.fit_transform(tensor)
print(means)

####################################
# Let's use the scikit-learn pipeline to chain these steps into a single
# decoding pipeline

from sklearn.pipeline import Pipeline
from sklearn.neighbors import KNeighborsClassifier

pipeline = Pipeline([
    ('spike_sampler',PeriEventSpikeSampler(spikes=spikes,sample_times=np.arange(0.1,0.35,0.01))),
    ('extract', ResponseReducer(func=np.mean)),
    ('classify', KNeighborsClassifier()),
])

####################################
# Now we can train the full pipeline on the training set

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    events, events['image_id'],test_size=0.33,
)

pipeline.fit(X_train,y_train)

####################################
# Finally we'll test the pipeline on the held out data

score = pipeline.score(X_test,y_test)

n_images = len(events['image_id'].unique())
print(score*n_images)
