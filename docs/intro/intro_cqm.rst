.. _intro_cqm:

=============================
Quadratic Models: Constrained
=============================

Many real-world problems include constraints. For example, a routing problem
might limit the number of airplanes on the ground at an airport and a scheduling
problem might require a minimum interval between shifts.

When using unconstrained samplers to handle problems with constraints, you
typically formulate the constraints as penalties. Constrained models such as
:class:`~dimod.ConstrainedQuadraticModel` can support constraints by encoding both an
objective and its set of constraints, as models or in symbolic form.

Supported Models
================

dimod provides a :term:`constrained quadratic model` (CQM) class that encodes
a quadratic objective and possibly one or more quadratic equality and inequality constraints.

For an introduction to CQMs, see
:std:doc:`Constrained Quadratic Models <oceandocs:concepts/cqm>`.

For descriptions of the CQM class and its methods, see :ref:`cqm`.

Model Construction
==================

dimod provides a variety of model generators. These are especially useful for testing
code and learning.

Example: dimod CQM Generator
----------------------------

This example creates a CQM representing a
`knapsack problem <https://en.wikipedia.org/wiki/Knapsack_problem>`_ of ten
items.

>>> cqm = dimod.generators.random_knapsack(10)

Typically you construct a model when reformulating your problem, using such
techniques as those presented in D-Wave's system documentation's
:std:doc:`sysdocs_gettingstarted:doc_handbook`.

Example: Formulating a CQM
--------------------------

This example constructs a CQM from symbolic math, which is especially useful for
learning and testing with small CQMs.

>>> x = dimod.Binary('x')
>>> y = dimod.Integer('y')
>>> cqm = dimod.CQM()
>>> objective = cqm.set_objective(x+y)
>>> cqm.add_constraint(y <= 3) #doctest: +ELLIPSIS
'...'

For very large models, you might read the data from a file or construct from a NumPy
array.
