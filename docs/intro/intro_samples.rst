.. _intro_samples:

=======
Samples
=======

dimod provides a :class:`~dimod.SampleSet` class that contains, and enables you to
manipulate, samples.

For an introduction to returned solutions and samples, see
:std:doc:`Solutions <oceandocs:concepts/solutions>`. For all supported sampleset
methods, see :ref:`sampleset`.

Example: Sampleset Returned from a Sampler
==========================================

This example creates a sample set and then demonstrates some :class:`~dimod.SampleSet`
properties and methods.

>>> bqm = dimod.generators.random.ran_r(1, 7)
>>> sampler = dimod.ExactSolver()
>>> sampleset = sampler.sample(bqm)

Print the best solution's energy.

>>> print(sampleset.first.energy)      # doctest:+SKIP
-9.0

Print the best solutions.

>>> print(sampleset.lowest())           # doctest:+SKIP
   0  1  2  3  4  5  6 energy num_oc.
0 +1 -1 +1 -1 +1 -1 -1   -9.0       1
1 +1 +1 +1 -1 +1 -1 -1   -9.0       1
...
7 -1 +1 -1 +1 -1 -1 +1   -9.0       1
['SPIN', 8 rows, 8 samples, 7 variables]

Convert to a third-party format
(`pandas <https://pandas.pydata.org/pandas-docs/stable/index.html>`_).

>>> sampleset.to_pandas_dataframe()       # doctest:+SKIP
     0  1  2  3  4  5  6  energy  num_occurrences
0   -1 -1 -1 -1 -1 -1 -1     3.0                1
1    1 -1 -1 -1 -1 -1 -1     7.0                1
2    1  1 -1 -1 -1 -1 -1     7.0                1
...

Example: Creating a Sampleset
=============================

This example creates a sample set from NumPy arrays.

>>> import numpy as np
>>> samples = np.random.randint(0, 2, (100, 10))
>>> energies = np.random.randint(-10, 0, 100)
>>> occurrences = np.random.randint(0, 50, 100)
>>> sampleset = dimod.SampleSet.from_samples(samples,
...                                          "BINARY",
...                                          energies,
...                                          num_occurrences=occurrences)
