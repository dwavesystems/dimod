.. image:: https://img.shields.io/pypi/v/dimod.svg
    :target: https://pypi.org/project/dimod

.. image:: https://codecov.io/gh/dwavesystems/dimod/branch/main/graph/badge.svg
    :target: https://codecov.io/gh/dwavesystems/dimod

.. image:: https://readthedocs.com/projects/d-wave-systems-dimod/badge/?version=latest
    :target: https://docs.ocean.dwavesys.com/projects/dimod/en/latest/?badge=latest

.. image:: https://circleci.com/gh/dwavesystems/dimod.svg?style=svg
    :target: https://circleci.com/gh/dwavesystems/dimod

dimod
=====

.. index-start-marker1

`dimod` is a shared API for samplers. It provides:

* a binary quadratic model (BQM) class that contains Ising and quadratic
  unconstrained binary optimization (QUBO) models used by samplers such as the
  D-Wave system.
* a discrete quadratic model (DQM) class and higher-order (non-quadratic) models.
* reference examples of samplers and composed samplers.
* `abstract base classes <https://docs.python.org/3/library/abc.html>`_ for
  constructing new samplers and composed samplers.


.. index-end-marker1

(For explanations of the terminology, see the
`Ocean glossary <https://docs.ocean.dwavesys.com/en/stable/concepts/index.html>`_.)

Example Usage
-------------

.. index-start-marker2

>>> import dimod
...
>>> # Construct a problem
>>> bqm = dimod.BinaryQuadraticModel({0: -1, 1: 1}, {(0, 1): 2}, 0.0, dimod.BINARY)
...
>>> # Use dimod's brute force solver to solve the problem
>>> sampleset = dimod.ExactSolver().sample(bqm)
>>> print(sampleset)
   0  1 energy num_oc.
1  1  0   -1.0       1
0  0  0    0.0       1
3  0  1    1.0       1
2  1  1    2.0       1
['BINARY', 4 rows, 4 samples, 2 variables]

.. index-end-marker2

See the documentation for more examples.

Installation
------------

.. installation-start-marker

Compatible with Python 3.5+:

.. code-block:: bash

    pip install dimod

To install with optional components:

.. code-block:: bash

    pip install dimod[all]

To install from source:

.. code-block:: bash

    pip install -r requirements.txt
    python setup.py install

When developing on dimod, it is often convenient to build the extensions
in place:

.. code-block:: bash

    pip install -r requirements.txt
    python setup.py build_ext --inplace

.. installation-end-marker

License
-------

Released under the Apache License 2.0. See LICENSE file.
