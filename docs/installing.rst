.. _installing:

.. highlight:: shell

============
Installation
============


Stable release
--------------

To install neuroglia, run this command in your terminal:

.. code-block:: console

    $ pip install git+https://github.com/AllenInstitute/neuroglia.git

To use the calcium module, including CalciumDeconvolver, you must also install OASIS:

.. code-block:: console

    $ pip install cython
    $ pip install git+https://github.com/j-friedrich/OASIS.git


From sources
------------

The sources for neuroglia can be downloaded from the `Github repo`_.

.. code-block:: console

    $ git clone git://github.com/AllenInstitute/neuroglia
    $ cd neuroglia
    $ pip install ./
