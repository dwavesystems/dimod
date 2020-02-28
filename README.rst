.. image:: https://img.shields.io/pypi/v/dimod.svg
    :target: https://pypi.python.org/pypi/dimod

.. image:: https://codecov.io/gh/dwavesystems/dimod/branch/master/graph/badge.svg
  :target: https://codecov.io/gh/dwavesystems/dimod

.. image:: https://readthedocs.com/projects/d-wave-systems-dimod/badge/?version=latest
  :target: https://docs.ocean.dwavesys.com/projects/dimod/en/latest/?badge=latest

.. image:: https://circleci.com/gh/dwavesystems/dimod.svg?style=svg
    :target: https://circleci.com/gh/dwavesystems/dimod

dimod
=====

.. index-start-marker1

`dimod` is a shared API for binary quadratic samplers. It provides a binary quadratic
model (BQM) class that contains Ising and quadratic unconstrained binary
optimization (QUBO) models used by samplers such as the D-Wave system. It also
provides utilities for constructing new samplers and composed samplers and for
minor-embedding. Its reference examples include several samplers and composed
samplers.

.. index-end-marker1

Learn more about `dimod on Read the Docs <https://docs.ocean.dwavesys.com/projects/dimod/en/latest/>`_\ .

Example Usage
-------------

.. index-start-marker2

This example constructs a simple QUBO and converts it to Ising format.

>>> import dimod
>>> bqm = dimod.BinaryQuadraticModel({0: -1, 1: -1}, {(0, 1): 2}, 0.0, dimod.BINARY)  # QUBO
>>> bqm_ising = bqm.change_vartype(dimod.SPIN, inplace=False)  # Ising

This example uses one of dimod's test samplers, ExactSolver, a solver that calculates
the energies of all possible samples.

>>> import dimod
>>> h = {0: 0.0, 1: 0.0}
>>> J = {(0, 1): -1.0}
>>> bqm = dimod.BinaryQuadraticModel.from_ising(h, J)
>>> response = dimod.ExactSolver().sample(bqm)
>>> for sample, energy in response.data(['sample', 'energy']): print(sample, energy)
{0: -1, 1: -1} -1.0
{0: 1, 1: 1} -1.0
{0: 1, 1: -1} 1.0
{0: -1, 1: 1} 1.0

.. index-end-marker2

See the documentation for more examples.

Installation
------------

.. installation-start-marker

Compatible with 3.5+:

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

Note that installation from source requires that your system have the Boost_
C++ libraries installed.

.. _Boost: https://www.boost.org/

.. installation-end-marker

License
-------

Released under the Apache License 2.0. See LICENSE file.

Contribution
------------

See CONTRIBUTING.rst file.
