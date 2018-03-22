.. image:: https://travis-ci.org/dwavesystems/dimod.svg?branch=master
    :target: https://travis-ci.org/dwavesystems/dimod

.. image:: https://ci.appveyor.com/api/projects/status/2oc8vrxxh15ecgo1?svg=true
    :target: https://ci.appveyor.com/project/dwave-adtt/dimod

.. image:: https://coveralls.io/repos/github/dwavesystems/dimod/badge.svg?branch=master
    :target: https://coveralls.io/github/dwavesystems/dimod?branch=master

.. image:: https://readthedocs.org/projects/dimod/badge/?version=latest
    :target: http://dimod.readthedocs.io/en/latest/?badge=latest

.. index-start-marker

dimod
=====

dimod is a shared API for binary quadratic samplers. It provides a binary quadratic
model (BQM) class that contains Ising and quadratic unconstrained binary
optimization (QUBO) models used by samplers such as the D-Wave system. It also
provides utilities for constructing new samplers and composed samplers.


Example Usage
-------------
This example constructs a simple QUBO and converts it to Ising format.

>>> import dimod
>>> bqm = dimod.BinaryQuadraticModel({0: -1, 1: -1}, {(0, 1): 2}, 0.0, dimod.BINARY)  # QUBO
>>> bqm_ising = bqm.change_vartype(dimod.SPIN, inplace=False)  # Ising

An example of using one of the built-in test Samplers.

>>> import dimod
>>> h = {0: 0.0, 1: 0.0}
>>> J = {(0, 1): -1.0}
>>> bqm = dimod.BinaryQuadraticModel.from_ising(h, J)
>>> response = dimod.ExactSolver().sample(bqm)
>>> response.samples_matrix
matrix([[-1, -1],
        [ 1, -1],
        [ 1,  1],
        [-1,  1]])

.. index-end-marker

See the documentation for more examples.

Installation
------------

.. installation-start-marker

Compatible with Python 2 and 3:

.. code-block:: bash

    pip install dimod

To install with optional components:

.. code-block:: bash

    pip install dimod[all]

To install from source:

.. code-block:: bash

    pip install -r requirements.txt
    python setup.py install

.. installation-end-marker

License
-------

Released under the Apache License 2.0. See LICENSE file.

Contribution
------------

See CONTRIBUTING.rst file.
