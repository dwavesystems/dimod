.. image:: https://img.shields.io/pypi/v/dimod.svg?style=svg
    :target: https://pypi.org/project/dimod

.. image:: https://img.shields.io/pypi/pyversions/dimod.svg?style=svg
    :target: https://pypi.python.org/pypi/dimod

.. image:: https://circleci.com/gh/dwavesystems/dimod.svg?style=svg
    :target: https://circleci.com/gh/dwavesystems/dimod

=====
dimod
=====

.. start_dimod_about

`dimod` is a shared API for samplers. It provides:

*   Classes for quadratic models---such as the binary quadratic model (BQM)
    class that contains Ising and QUBO models used by samplers such as the
    D-Wave quantum computer---and higher-order (non-quadratic) models.
*   Reference examples of samplers and composed samplers.
*   `Abstract base classes <https://docs.python.org/3/library/abc.html>`_ for
    constructing new samplers and composed samplers.

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

.. end_dimod_about

For explanations of the terminology, see the
`Ocean glossary <https://docs.dwavequantum.com/en/latest/concepts/index.html>`_.

See the `documentation <https://docs.dwavequantum.com/en/latest/index.html>`_
for more examples.

Installation
============

Installation from `PyPI <https://pypi.org/project/dimod>`_:

.. code-block:: bash

    pip install dimod

During package development, it is often convenient to use an editable install.
See `meson-python's editable installs
<https://meson-python.readthedocs.io/en/latest/how-to-guides/editable-installs.html>`_
for more details.

.. code-block:: bash

    pip install --group dev
    pip install --editable . \
        --no-build-isolation \
        --config-settings=editable-verbose=true

Testing
=======

All code should be thoroughly tested and all pull requests should include tests.

To run the Python tests, first install the package using an editable install
as described above. You will also need to install additional dependencies for
the tests.

.. code-block:: bash

    pip install --group tests

The tests can then be run with
`unittest <https://docs.python.org/3/library/unittest.html>`_.

.. code-block:: bash

    python -m unittest

To run the C++ tests, first install the project dependencies, then setup a
``meson`` build directory. You must configure the build as a debug build for
the tests to run.

.. code-block:: bash

    pip install --group dev
    meson setup build -Dbuildtype=debug

You can then run the tests using
`meson's test framework <https://mesonbuild.com/Unit-tests.html>`_.

.. code-block:: bash

    meson test -Cbuild
