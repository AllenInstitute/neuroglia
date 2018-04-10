#!/usr/bin/env python
"""
natural scene decoding ophys
==============================

This is an example of how to decode natural images from ophys traces in V1

"""

OEID = 541206592

#######################################
# First, let's download an experiment from the Allen Institute Brain Observatory

from allensdk.core.brain_observatory_cache import BrainObservatoryCache
nwb_dataset = boc.get_ophys_experiment_data(OEID)

#############################################
# Next, we'll load the dF/F traces and put them in a DataFrame

timestamps, dff = nwb_dataset.get_dff_traces()
neuron_ids = nwb_dataset.get_cell_specimen_ids()

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

stim_table['time'] = timestamps[stim_table['Start']]
stim_tabel['End']  = timestamps[stim_table['End']]

stim_table['duration'] = stim_tabel['End'] - stim_tabel['time']

print(stim_table.head())
