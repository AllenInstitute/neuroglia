#!/usr/bin/env python
"""
Decode location from CA1 activity
==============================

This is an example of how to decode location in a free field from CA1 activity

"""

from neuroglia.datasets import fetch_rat_hippocampus_foraging

dataset = fetch_rat_hippocampus_foraging()

#########################################
# Let's plot the path in the free field

import matplotlib.pyplot as plt
plt.plot(dataset.location['x'], dataset.location['y'], alpha=0.5)
plt.axis('equal')
plt.show()

#########################################
# Create a feature vector for each time point in the location data

from neuroglia.spike import Binner

binner = Binner(sample_times=dataset.location['time'])
response = binner.fit_transform(dataset.spikes)
print(response.head())

#########################################
# create feature


from sklearn.linear_model import LinearRegression
lm = LinearRegression()

split = int(len(response)/2)

X_train = response.values[:split]
y_train = dataset.location['x'].values[:split]

lm.fit(X_train,y_train)


X_test = response.values[split:]
y_test = dataset.location['x'].values[split:-1]
y_pred = lm.predict(X_test)
plt.plot(y_test[:100])
plt.plot(y_pred[:100])
plt.show()
