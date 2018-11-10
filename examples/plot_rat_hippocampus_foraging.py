#!/usr/bin/env python
"""
Dataset: CA1 activity during foraging
==============================

This is an example of how to access data recorded from CA1 during open field foraging

"""

from neuroglia.datasets import fetch_rat_hippocampus_foraging

dataset = fetch_rat_hippocampus_foraging()

#########################################
# Let's plot the path in the free field

import matplotlib.pyplot as plt
plt.plot(dataset.location['x'], dataset.location['y'])
plt.axis('equal')
plt.show()

#########################################
# Create a feature vector, binning spikes for each time point

from neuroglia.spike import Binner

binner = Binner(sample_times=dataset.location['time'])
response = binner.fit_transform(dataset.spikes)

#########################################
# Plot CA1 activity

response.plot()
plt.show()
