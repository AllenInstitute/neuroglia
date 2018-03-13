#!/usr/bin/env python
"""
Deconvolve synthetic calcium traces
==============================

This is an example of how to infer spike events

"""

#######################################################
# First, we'll generate some fake data

import numpy as np
import pandas as pd
from neuroglia.oasis.functions import gen_data

neuron_ids = ['a', 'b', 'c']
sampling_rate = 30.0

traces, _, spikes = map(np.squeeze, gen_data(N=3, b=2, seed=0))

time = np.arange(0, traces.shape[1]/sampling_rate, 1/sampling_rate)

traces = pd.DataFrame(traces.T, index=time, columns=neuron_ids)
spikes = pd.DataFrame(spikes.T, index=time, columns=neuron_ids)

########################################################
# let's plot the data

import matplotlib.pyplot as plt
traces.plot()
plt.show()

##########################################################
# Now, we'll deconvolve the data

from neuroglia.calcium import CalciumDeconvolver

deconvolver = CalciumDeconvolver()
detected_events = deconvolver.transform(traces)

for neuron in neuron_ids:
    y_true = spikes[neuron]
    y_pred = detected_events[neuron]
    corr = np.corrcoef(y_pred,y_true)[0,1]
    print("{}: {:0.2f}".format(neuron,corr))

detected_events.plot()
plt.show()

##########################################################
# Now, we'll predict spikes

spikes_pred = deconvolver.predict(traces)
spikes_true = (spikes>0).astype(int)

for neuron in neuron_ids:
    y_true = spikes_true[neuron]
    y_pred = spikes_pred[neuron]
    corr = np.corrcoef(y_pred,y_true)[0,1]
    print("{}: {:0.2f}".format(neuron,corr))
