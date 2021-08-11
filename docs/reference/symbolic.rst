
.. _symbolic_math:

=============
Symbolic Math
=============

You can construct a model, for example a CQM, from symbolic math, which is especially
useful for learning and testing with small problems.

dimod enables easy incorporation of binary and integer variables as single-variable
models. For example, you can represent such binary variables as follows:

>>> from dimod import Binary, Spin
>>> x = Binary('x')
>>> s = Spin('s')
>>> x
BinaryQuadraticModel({'x': 1.0}, {}, 0.0, 'BINARY')

Similarly for integers:

>>> from dimod import Integer
>>> i = Integer('i')
>>> i
QuadraticModel({'i': 1.0}, {}, 0.0, {'i': 'INTEGER'}, dtype='float64')

The construction of such variables as either BQMs or QMS depends on the type of
variable:

>>> x + s
QuadraticModel({'x': 1.0, 's': 1.0}, {}, 0.0, {'x': 'BINARY', 's': 'SPIN'}, dtype='float64')
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
