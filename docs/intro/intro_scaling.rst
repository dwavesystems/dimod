.. _intro_scaling:

======================
Scaling for Production
======================

This tutorial is aimed at users who wish to scale their application problems to
the industrial scale supported by the Leap Hybrid Solvers.

This guide will focus on :ref:`intro_cqm` though most of the concept are
applicable to :ref:`Binary Quadratic Models (BQMs) <intro_qm_bqm>`
and :ref:`Quadratic Models (QMs) <intro_qm_qm>`.

This guide does not discuss algorithmic complexity or problem formulation.
For those topics, see the :doc:`sysdocs_gettingstarted:doc_getting_started`.


Your first application
======================

.. tip::

    .. dev note: in the future we should consider using nbsphinx or similar
        for this. But as of now (April 2022) nbsphinx is a bit immature for
        our needs. E.g. has non-pip-installable requirements, doesn't play
        nicely with intersphinx, etc.

    You can easily run the code for this tutorial by downloading the
    :download:`Jupyter Notebook <intro_scaling.ipynb>`.

Let us construct a simple `bin packing <https://w.wiki/3jz4>`_ problem.
We assume that each bin has a capacity of ``1``.

We start by generating weights for the items we wish to pack.
A packing problem with ``n`` items will result in a  CQM with ``n*(n+1)`` binary variables.

.. code-block:: python

    import numpy as np

    num_items = 100  # results in 10100 binary variables

    weights = np.random.default_rng(42).random(num_items)

The first implementation is optimized for readability and pedagogy.
Though as we will see, this comes at the cost of speed.

.. code-block:: python

    import typing

    import dimod


    def bin_packing(weights: typing.Sequence[float]) -> dimod.ConstrainedQuadraticModel:
        """Generate a bin packing problem as a constrained quadratic model."""

        n = len(weights)
        
        # y_j indicates that bin j is used
        y = [dimod.Binary(f'y_{j}') for j in range(n)]
        
        # x_i,j indicates that item i is put in bin j
        x = [[dimod.Binary(f'x_{i},{j}') for j in range(n)] for i in range(n)]
        
        cqm = dimod.ConstrainedQuadraticModel()
        
        # we wish to minimize the number of bins used
        cqm.set_objective(sum(y))
        
        # each item can only go in one bin
        for i in range(n):
            cqm.add_constraint(sum(x[i]) == 1, label=f'item_placing_{i}')
            
        # each bin has a capacity that must be respected
        for j in range(n):
            cqm.add_constraint(sum(weights[i] * x[i][j] for i in range(n)) - y[j] <= 0,
                               label=f'capacity_bin_{j}')
            
        return cqm

Let's see how long the construction takes.

.. code-block:: ipythonconsole

    In [1]: %timeit bin_packing(weights)
    385 ms ± 9.8 ms per loop (mean ± std. dev. of 7 runs, 1 loop each)

.. note::

    Runtimes are highly system dependent. The numbers here are meant to be
    representative. You may get different values when you run them on your
    own system.

Use quicksum
============

The first and easiest change we wan make is to use :func:`~dimod.binary.quicksum`
as a replacement for the Python :func:`sum`.
Python's :func:`sum` creates a large number of intermediate objects, whereas
:func:`~dimod.binary.quicksum` does not.

.. code-block:: python

    import typing

    import dimod


    def bin_packing(weights: typing.Sequence[float]) -> dimod.ConstrainedQuadraticModel:
        """Generate a bin packing problem as a constrained quadratic model."""

        n = len(weights)
        
        # y_j indicates that bin j is used
        y = [dimod.Binary(f'y_{j}') for j in range(n)]
        
        # x_i,j indicates that item i is put in bin j
        x = [[dimod.Binary(f'x_{i},{j}') for j in range(n)] for i in range(n)]
        
        cqm = dimod.ConstrainedQuadraticModel()
        
        # we wish to minimize the number of bins used
        cqm.set_objective(dimod.quicksum(y))
        
        # each item can only go in one bin
        for i in range(n):
            cqm.add_constraint(dimod.quicksum(x[i]) == 1, label=f'item_placing_{i}')
            
        # each bin has a capacity that must be respected
        for j in range(n):
            cqm.add_constraint(dimod.quicksum(weights[i] * x[i][j] for i in range(n)) - y[j] <= 0,
                               label=f'capacity_bin_{j}')
            
        return cqm

This results in some time savings.

.. code-block:: ipythonconsole

    In [1]: %timeit bin_packing(weights)
    294 ms ± 9.39 ms per loop (mean ± std. dev. of 7 runs, 1 loop each)

Construct the models individually
=================================

Although :func:`~dimod.binary.quicksum` improves the performance, we can get an
even bigger improvement by skipping symbolic construction altogether.
See :ref:`Symbolic Math <intro_symbolic_math>` for a discussion of the
difference between variables and labels.

We can demonstrate the performance difference with a small example.

.. code-block:: python

    import dimod

    def make_bqm_symbolic(num_variables: int) -> dimod.BinaryQuadraticModel:
        return dimod.quicksum(2*dimod.Binary(v) for v in range(num_variables))

    def make_bqm_labels(num_variables: int) -> dimod.BinaryQuadraticModel:
        bqm = dimod.BinaryQuadraticModel('BINARY')
        bqm.add_linear_from((v, 2) for v in range(num_variables))
        return bqm

Working directly with the variable labels and a single BQM object gives a significant speedup

.. code-block:: ipythonconsole

    In [1]: %timeit make_bqm_symbolic(1000)
    12.7 ms ± 213 µs per loop (mean ± std. dev. of 7 runs, 100 loops each)
    In [2]: %timeit make_bqm_labels(1000)
    194 µs ± 2.32 µs per loop (mean ± std. dev. of 7 runs, 10000 loops each)

Let's apply the construction by labels to our binpacking example

.. code-block:: python

    import typing

    import dimod


    def bin_packing(weights: typing.Sequence[float]) -> dimod.ConstrainedQuadraticModel:
        """Generate a bin packing problem as a constrained quadratic model."""

        n = len(weights)
        
        # y_j indicates that bin j is used
        y_labels = [f'y_{j}' for j in range(n)]
        
        # x_i,j indicates that item i is put in bin j
        x_labels = [[f'x_{i},{j}' for j in range(n)] for i in range(n)]
        
        cqm = dimod.ConstrainedQuadraticModel()
        
        # we wish to minimize the number of bins used
        objective = dimod.QuadraticModel()
        objective.add_linear_from(((v, 1) for v in y_labels), default_vartype='BINARY')
        cqm.set_objective(objective)
        
        # each item can only go in one bin
        for i in range(n):
            lhs = dimod.QuadraticModel()
            lhs.add_linear_from(((v, 1) for v in x_labels[i]), default_vartype='BINARY')
            cqm.add_constraint_from_model(lhs, rhs=1, sense='==', label=f'item_placing_{i}')
            
        # each bin has a capacity that must be respected
        for j in range(n):
            lhs = dimod.QuadraticModel()
            lhs.add_linear_from(((x_labels[i][j], weights[i]) for i in range(n)), default_vartype='BINARY')
            lhs.add_linear(y_labels[j], -1, default_vartype='BINARY')
            cqm.add_constraint_from_model(lhs, rhs=0, sense='<=', label=f'capacity_bin_{j}')
            
        return cqm

This gives us significant time savings

.. code-block:: ipythonconsole

    In [1]: %timeit bin_packing(weights)
    95.5 ms ± 2.87 ms per loop (mean ± std. dev. of 7 runs, 10 loops each)

Don't copy constraints
======================

By default :meth:`~dimod.ConstrainedQuadraticModel.add_constraint`
create a copy of the objects given to it.
However, in this case we are immediately discarding the models created in our
function, so we can safely skip the copy step.

.. code-block:: python

    import typing

    import dimod


    def bin_packing(weights: typing.Sequence[float]) -> dimod.ConstrainedQuadraticModel:
        """Generate a bin packing problem as a constrained quadratic model."""

        n = len(weights)
        
        # y_j indicates that bin j is used
        y_labels = [f'y_{j}' for j in range(n)]
        
        # x_i,j indicates that item i is put in bin j
        x_labels = [[f'x_{i},{j}' for j in range(n)] for i in range(n)]
        
        cqm = dimod.ConstrainedQuadraticModel()
        
        # we wish to minimize the number of bins used
        objective = dimod.QuadraticModel()
        objective.add_linear_from(((v, 1) for v in y_labels), default_vartype='BINARY')
        cqm.set_objective(objective)
        
        # each item can only go in one bin
        for i in range(n):
            lhs = dimod.QuadraticModel()
            lhs.add_linear_from(((v, 1) for v in x_labels[i]), default_vartype='BINARY')
            cqm.add_constraint_from_model(lhs, rhs=1, sense='==', label=f'item_placing_{i}', copy=False)
            
        # each bin has a capacity that must be respected
        for j in range(n):
            lhs = dimod.QuadraticModel()
            lhs.add_linear_from(((x_labels[i][j], weights[i]) for i in range(n)), default_vartype='BINARY')
            lhs.add_linear(y_labels[j], -1, default_vartype='BINARY')
            cqm.add_constraint_from_model(lhs, rhs=0, sense='<=', label=f'capacity_bin_{j}', copy=False)
            
        return cqm

This results in another performance improvement.

.. code-block:: ipythonconsole

    In [1]: %timeit bin_packing(weights)
    68.1 ms ± 299 µs per loop (mean ± std. dev. of 7 runs, 10 loops each)
