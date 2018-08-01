#!/usr/bin/env python
"""
Create synthetic calcium traces
==============================

This is an example of how to create synthetic calcium traces

"""


from neuroglia.datasets import make_calcium_traces

calcium = make_calcium_traces(duration=10.0,oscillation=False)
calcium['traces'].plot()

########################################################
# The synthetic spikes that underly the synthetic calcium traces are also
# available

calcium['spikes'].plot()

########################################################
# We can also generate synthetic calcium traces where a gamma oscillation
# provides an input to the population

calcium = make_calcium_traces(duration=10.0,oscillation=True)
calcium['traces'].plot()
