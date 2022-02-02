.. _intro_symbolic_math:

=============
Symbolic Math
=============

dimod enables easy incorporation of binary and integer variables as
:ref:`single-variable models <symbolic_math_generators>`. For example, you can
represent such binary variables as follows:

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

The construction of such variables as either a
:class:`~dimod.binary.binary_quadratic_model.BinaryQuadraticModel` or a
:class:`~dimod.quadratic.quadratic_model.QuadraticModel` depends on the type of
variable. Models with more than one type of variable, for example
:func:`~dimod.binary.Binary` and :func:`~dimod.binary.Spin`, or one of those
with :func:`~dimod.quadratic.Integer`, are of the
:class:`~dimod.quadratic.quadratic_model.QuadraticModel` class.

>>> z = x + s
>>> print("Type of {} is {}".format(z.to_polystring(), type(z)))
Type of x + s is <class 'dimod.quadratic.quadratic_model.QuadraticModel'>
>>> for variable in z.variables:
...     print("{} is of type {}.".format(variable, z.vartype(variable)))
x is of type Vartype.BINARY.
s is of type Vartype.SPIN.

You can express mathematical functions on these variables using Python functions such
as :func:`sum`\ [#]_\ :

.. [#]
  See the `Example: Adding Models`_ example for a performant summing function.

>>> sum([3 * i, 2 * i]).to_polystring()
'5*i'

.. note::
  It's important to remember that, for example, :code:`x = dimod.Binary('x')`
  instantiates a single-variable model, in this case a
  :class:`~dimod.binary.binary_quadratic_model.BinaryQuadraticModel` with
  variable label ``'x'``, not a free-floating variable labeled ``x``. Consequently,
  you can add ``x`` to another model, say :code:`bqm = dimod.BinaryQuadraticModel('BINARY')`,
  by adding the two models, :code:`x + bqm`. This adds the variable labeled ``'x'``
  in the single-variable BQM, ``x`` to model ``bqm``. You cannot add ``x`` to a
  model---as though it were variable ``'x'``---by doing :code:`bqm.add_variable(x)`.

Example: BQM
============

This example creates the BQM :math:`x + 2y -xy`:

>>> from dimod import Binary
>>> x = Binary('x')
>>> y = Binary('y')
>>> bqm = x + 2*y - x*y
>>> print(bqm.to_polystring())
x + 2*y - x*y

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

Example: Adding Models
======================

This example uses the performant :func:`~dimod.binary.quicksum` on
:func:`~dimod.binary.BinaryArray` to add multiple models.

>>> import numpy as np
>>> from dimod import BinaryArray, quicksum
...
>>> num_vars = 10; max_bias = 5
>>> var_labels = range(num_vars)
...
>>> models = BinaryArray(var_labels)*np.random.randint(0, max_bias, size=num_vars)
>>> x = quicksum(models)
