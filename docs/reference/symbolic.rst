
.. _symbolic_math:

=============
Symbolic Math
=============

You can construct a model, for example a CQM, from symbolic math, which is especially
useful for learning and testing with small problems.

dimod supports binary and integer variables:

>>> from dimod import Binary, Integer
>>> # Create binary variables
>>> x = Binary('x')
>>> y = Binary('y')
>>> # Create integer variables
>>> i = Integer('i')
>>> j = Integer('j')

Such variables are constructed as either BQMs or QMS, depending on the type of
variable:

>>> 2*x
BinaryQuadraticModel({'x': 2.0}, {}, 0.0, 'BINARY')
>>> 3*i - x
QuadraticModel({'i': 3.0, 'x': -1.0}, {}, -0.0, {'i': 'INTEGER', 'x': 'BINARY'}, dtype='float64')

You can express mathematical functions on these variables using Python functions such
as :func:`sum`:

>>> sum([3*i, 2*i])
QuadraticModel({'i': 5.0}, {}, 0.0, {'i': 'INTEGER'}, dtype='float64')

Example: BQM
============

This example creates the BQM :math:`x + 2y -xy`:

>>> from dimod import Binary
>>> x = Binary('x')
>>> y = Binary('y')
>>> bqm = x + 2*y - x*y

Example: CQM
============

This example uses symbolic math to set an objective (:math:`2i - 0.5ij + 10`)
and constraints (:math:`xj <= 3` and :math:`i + j >= 1`) in a simple CQM.

>>> from dimod import Binary, Integer, ConstrainedQuadraticModel
>>> x = Binary('x')
>>> i = Integer('i')
>>> j = Integer('j')
>>> cqm = ConstrainedQuadraticModel()
>>> cqm.set_objective(2*i - 0.5*i*j + 10)
>>> cqm.add_constraint(x*j <= 3)                   # doctest: +IGNORE_RESULT
>>> cqm.add_constraint(i + j >= 1)                 # doctest: +IGNORE_RESULT

.. currentmodule:: dimod.sym

Class
=====

.. autoclass:: Sense
