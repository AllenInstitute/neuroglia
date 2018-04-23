.. _api_ref:

.. currentmodule:: neuroglia

API reference
=============

.. _spike_api:

Spike transformers
------------------

.. autosummary::
   :toctree: generated/

    spike.Binner
    spike.Smoother
    nwb.SpikeTablizer

.. _trace_api:

Trace transformers
------------------

.. autosummary::
   :toctree: generated/

    trace.Binarizer
    trace.EdgeDetector
    trace.WhenTrueFinder


.. _calcium_api:

Calcium transformers
--------------------

.. autosummary::
   :toctree: generated/
    calcium.CalciumDeconvolver
    calcium.MedianFilterDetrender
    calcium.SavGolFilterDetrender
    calcium.EventRescaler


.. _event_api:

Event transformers
------------------

.. autosummary::
   :toctree: generated/

    event.PeriEventSpikeSampler
    event.PeriEventTraceSampler
    event.PeriEventTraceReducer

.. _epoch_api:

Epoch transformers
------------------

.. autosummary::
   :toctree: generated/

    epoch.EpochTraceReducer

.. _tensor_api:

Tensor transformers
-------------------

.. autosummary::
   :toctree: generated/

    tensor.ResponseReducer
