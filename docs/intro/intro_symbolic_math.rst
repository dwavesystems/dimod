.. _intro_symbolic_math:

=============
Symbolic Math
=============

*dimod*'s support for symbolic math can simplify your coding of problems. For example,
a problem of finding the rectangle with the greatest area for a given perimeter,
:math:`P = 8`,

.. math::

  \max_{i,j} \quad ij

  \textrm{s.t.} \quad 2i+2j \le P

formulated as an :term:`objective`---maximize\ [#]_ :math:`ij`, the multiplication
of side :math:`i` by side :math:`j`---subject to the :term:`constraint` that the
summation of the rectangle's four sides not exceed the perimeter,
:math:`2i+2j \le P`, can be represented as such,

>>> print(objective.to_polystring())            # doctest:+SKIP
-i*j
>>> print(constraint.lhs.to_polystring(), constraint.sense.value, constraint.rhs)  # doctest:+SKIP
2*i + 2*j <= 8

.. [#] The coded ``objective`` is set to negative because D-Wave samplers minimize
  rather than maximize.

The foundation for this symbolic representation is single-variable models.

Variables as Models
===================

To symbolically represent an objective or constraint, you first need symbolic
representations of variables. In problems such as that of the example above, the
type of variable needed might be integer:

>>> from dimod import Integer
>>> i = Integer('i')
>>> i
QuadraticModel({'i': 1.0}, {}, 0.0, {'i': 'INTEGER'}, dtype='float64')

Such a variable is represented by one of dimod's supported quadratic models with a
single variable; here, variable ``i`` is a
:class:`~dimod.quadratic.quadratic_model.QuadraticModel` with one variable with
the label ``'i'``. This works because quadratic models are problems of the form,

.. math::

    \sum_i a_i x_i + \sum_{i<j} b_{i, j} x_i x_j + c

where :math:`\{ x_i\}_{i=1, \dots, N}` can be binary or integer
variables and :math:`a_{i}, b_{ij}, c` are real values. If you set :math:`a_1=1`
and all remaining coefficients to zero, the model represents a single variable,
:math:`x_1`.

Similarly, a linear term, such as :math:`3.7i`, can be represented by this same
model by setting the appropriate linear coefficient on the ``'i'``--labeled variable:

>>> 3.75 * i
QuadraticModel({'i': 3.75}, {}, 0.0, {'i': 'INTEGER'}, dtype='float64')

And adding a non-zero quadratic coefficient, :math:`b_{11}`

>>> 2.2 * i * i + 3.75 * i
QuadraticModel({'i': 3.75}, {('i', 'i'): 2.2}, 0.0, {'i': 'INTEGER'}, dtype='float64')

, not a free-floating variable labeled ``x``. Consequently,
you can add ``x`` to another model, say :code:`bqm = dimod.BinaryQuadraticModel('BINARY')`,
by adding the two models, :code:`x + bqm`. This adds the variable labeled ``'x'``
in the single-variable BQM, ``x`` to model ``bqm``. You cannot add ``x`` to a
model---as though it were variable ``'x'``---by doing :code:`bqm.add_variable(x)`.

dimod supports various methods of creating

dimod enables easy incorporation of binary and integer variables as
:ref:`single-variable models <generators_symbolic_math>`. For example, you can
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
