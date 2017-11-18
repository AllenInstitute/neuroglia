.. _introduction:

Introduction
============

Neuroglia is a suite of scikit-learn transformers to facilitate converting between the canonical data structures used in ephys & ophys:

-	Spike times: a list of timestamps labelled with the neuron that elicited the spike
-	Traces: a 2D array of neurons x time. Aka a “time series”. E.g. calcium traces from 2P. binned spike times. Gaussian smoothed spike times, etc.
-	Tensor: a 3D array of traces aligned to events (events x neurons x time)

scikit-learn transformers
-------------------------

Transformations between these representations are implemented as scikit-learn transformers. This means that they all are defined as objects with “fit” and “transform” methods so that, for example, applying a Gaussian smoothing to a population of spiking data means transforming from a “spike times” structure to a “traces” structure like so:
::

    smoother = neuroglia.spike.Smoother(
        sample_times=np.arange(0,MAX_TIME,0.001), # <- this is the time base that the smoothed traces will be cast onto
        kernel=’gaussian’, # <- this is the kernel that will be used
        tau=0.005, # <- this is the width of the kernel in whatever time base the spike times are in
    )

    smoothed_traces = smoother.fit_transform(SPIKES)

Conforming to the syntax that is expected by the scikit learn API turns these transformers into building blocks that can plug into a scikit learn pipeline. For example, let’s say you wanted to do some dimensionality reduction on the smoothed traces.
::

    from sklearn.decomposition import NMF

    nmf = NMF(n_components=10)
    reduced_traces = nmf.fit_transform(smoothed_traces)

machine learning pipelines
--------------------------

You could also chain these together like so
::

    from sklearn.pipeline import Pipeline

    pipeline = Pipeline([
        (‘smooth’,smoother),
        (‘reduce’, nmf),
    ])

    reduced_traces = pipeline.fit_transform(SPIKES)

And if you wanted to change an analysis step, it just becomes a matter of replacing that piece of the pipeline
::

    from sklearn.decomposition import PCA

    pipeline = Pipeline([
        (‘smooth’,smoother),
        (‘reduce’, PCA(n_components=10)),
    ])

    reduced_traces = pipeline.fit_transform(SPIKES)


event-aligned responses
-----------------------

I’ve also implemented annotating events with event-aligned responses, so I can build an entire decoding pipeline that decodes the stimulus that was presented to a population from (for example)  the peak response in any 10ms bin in a 250ms window after the stimulus onset:
::

    from neuroglia.event import PeriEventSpikeSampler
    from neuroglia.tensor import ResponseReducer
    from sklearn.neighbors import KNeighborsClassifier

    pipeline = Pipeline([
        ('sample', PeriEventSpikeSampler(
            spikes=SPIKES,
            sample_times=np.arange(0.0,0.25,0.01),
            tracizer=Binner,
            )),
        ('reduce', ResponseReducer(method='max')),
        ('classify', KNeighborsClassifier()),
    ])

cross validation of an entire pipeline
--------------------------------------

Then, once this pipeline has defined, we can take advantage of scikit-learn infrastructure for cross validation to do a 4-fold cross validation across stimulus presentations
::

    from sklearn.model_selection import cross_val_score

    X = EVENTS[‘times’]
    y = EVENTS[‘image_id’]

    scores = cross_val_score(pipeline, X, y, cv=4)

These examples illustrate the major features of the package & how the API works.
