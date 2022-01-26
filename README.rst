.. image:: https://img.shields.io/pypi/v/dimod.svg
    :target: https://pypi.org/project/dimod

.. image:: https://img.shields.io/pypi/pyversions/dimod.svg
    :target: https://pypi.python.org/pypi/dimod

.. image:: https://circleci.com/gh/dwavesystems/dimod.svg?style=svg
    :target: https://circleci.com/gh/dwavesystems/dimod

.. image:: https://codecov.io/gh/dwavesystems/dimod/branch/main/graph/badge.svg
    :target: https://codecov.io/gh/dwavesystems/dimod

dimod
=====

.. index-start-marker1

`dimod` is a shared API for samplers. It provides:

* classes for quadratic models---such as the binary quadratic model (BQM) class that
  contains Ising and QUBO models used by samplers such as the D-Wave system---and
  higher-order (non-quadratic) models.
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

See the `documentation <https://docs.ocean.dwavesys.com/en/stable/docs_dimod/>`_
for more examples.

Installation
------------

.. installation-start-marker

Compatible with Python 3.6+:

.. code-block:: bash

    pip install dimod

To install from source (requires ``pip>=10.0.0``):

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

Contributing
------------

Ocean's `contributing guide <https://docs.ocean.dwavesys.com/en/stable/contributing.html>`_
has guidelines for contributing to Ocean packages.

dimod includes some formatting customization in the
`.clang-format <.clang-format>`_ and `setup.cfg <setup.cfg>`_ files.

Release Notes
~~~~~~~~~~~~~

dimod makes use of `reno <https://docs.openstack.org/reno/>`_ to manage its
release notes.

When making a contribution to dimod that will affect users, create a new
release note file by running

.. code-block:: bash

    reno new your-short-descriptor-here

You can then edit the file created under ``releasenotes/notes/``.
Remove any sections not relevant to your changes.
Commit the file along with your changes.

See reno's `user guide <https://docs.openstack.org/reno/latest/user/usage.html>`_
for details.
