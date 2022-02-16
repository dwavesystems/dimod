.. _intro_symbolic_math:

=============
Symbolic Math
=============

*dimod*'s support for symbolic math can simplify your coding of problems. For
example, consider a problem of finding the rectangle with the greatest area for
a given perimeter, which you can formulate mathematically as,

.. math::

  \max_{i,j} \quad ij

  \textrm{s.t.} \quad 2i+2j \le P

where the components are,

* **Variables**: :math:`i` and :math:`j` are the lengths of two sides and :math:`P`
  is the length of the perimeter.
* **Objective**: maximize the area, which is given by the standard geometric
  formula :math:`ij`.
* **Constraint**: subject to not exceeding the given perimeter length; that is,
  :math:`2i+2j`, the summation of the rectangle's four sides, is constrained to
  a maximum value of :math:`P`.

*dimod*'s symbolic math enables an intuitive coding of such problems:

>>> print(objective.to_polystring())            # doctest:+SKIP
-i*j
>>> print(constraint.lhs.to_polystring(), constraint.sense.value, constraint.rhs)  # doctest:+SKIP
2*i + 2*j <= 8

Here, the coded ``objective`` is set to negative because D-Wave samplers minimize
rather than maximize, and :math:`P` in the ``constraint`` was selected arbitrarily
to be 8.

The foundation for this symbolic representation is single-variable models.

Variables as Models
===================

To symbolically represent an objective or constraint, you first need symbolic
representations of variables.

You can use a quadratic model with a single variable to represent your
variable; for example, if the type of variable you need is integer:

>>> from dimod import Integer
>>> i = Integer('i')
>>> i
QuadraticModel({'i': 1.0}, {}, 0.0, {'i': 'INTEGER'}, dtype='float64')

Here, variable ``i`` is a :class:`~dimod.quadratic.quadratic_model.QuadraticModel`
with one variable with the label ``'i'``.

This works because quadratic models have the form,

.. math::

    \sum_i a_i x_i + \sum_{i \le j} b_{i, j} x_i x_j + c

where :math:`\{ x_i\}_{i=1, \dots, N}` can be integer variables
(:math:`a_{i}, b_{ij}, c` are real values). If you set :math:`a_1=1` and all
remaining coefficients to zero, the model represents a single variable,
:math:`x_1`.

When your variable is in a linear term of a polynomial, such as :math:`3.7i`,
the coefficient (:math:`3.7`) is represented in this same model by the value of
the linear bias on the ``'i'``--labeled variable:

>>> 3.75 * i
QuadraticModel({'i': 3.75}, {}, 0.0, {'i': 'INTEGER'}, dtype='float64')

Similarly, when your variable is in a quadratic term, such as :math:`2.2i^2`, the
coefficient (:math:`2.2`) is represented in this same model by the value of
the quadratic bias, :math:`b_{1, 1} = 2.2`, on the ``'i'``--labeled variable:

>>> (2.2 * i * i + 3.75 * i).quadratic
{('i', 'i'): 2.2}

You can see the various methods of creating variables in the
:ref:`reference documentation <generators_symbolic_math>`.

Typically, you have more than a single variable, and your variables interact.

Operations on Variables
=======================

Consider a simple problem of an AND operation on two binary variables. For
:math:`\{0, 1\}`--valued binary variables, the AND operation is equivalent to
the multiplication of the two variables:

>>> from dimod import Binaries, ExactSolver
>>> x, y = Binaries(["x", "y"])
>>> bqm_and = x*y
>>> bqm_and
BinaryQuadraticModel({'x': 0.0, 'y': 0.0}, {('y', 'x'): 1.0}, 0.0, 'BINARY')
>>> print(ExactSolver().sample(bqm_and))
   x  y energy num_oc.
0  0  0    0.0       1
1  1  0    0.0       1
3  0  1    0.0       1
2  1  1    1.0       1
['BINARY', 4 rows, 4 samples, 2 variables]

The symbolic multiplication between variables above implemented multiplication
between the models representing each variable. Binary quadratic models (BQMs) are
of the form:

  .. math::

      \sum_{i=1} a_i v_i
      + \sum_{i<j} b_{i,j} v_i v_j
      + c
      \qquad\qquad v_i \in\{-1,+1\} \text{  or } \{0,1\}

where :math:`a_{i}, b_{ij}, c` are real values. The multiplication of two such
models, with linear terms :math:`a_1 = 1`, reduced to
:math:`\sum_{i=1} 1 x_1 * \sum_{i=1} 1 y_1 = x_1y_1`.

Here, because all the variables are the same :class:`~dimod.Vartype`,
:class:`~dimod.Vartype.BINARY`, dimod instantiates a
:class:`~dimod.binary.binary_quadratic_model.BinaryQuadraticModel` to represent
each binary variable.

>>> bqm_and.vartype == dimod.Vartype.BINARY
True

If an operation includes more than one type of variable, the representation is
always a :class:`~dimod.quadratic.quadratic_model.QuadraticModel` and the
:class:`~dimod.Vartype` is per variable:

>>> print(type(bqm_and + 3.75 * i))
<class 'dimod.quadratic.quadratic_model.QuadraticModel'>
>>> (bqm_and + 3.75 * i).vartype("x") == dimod.Vartype.BINARY
True
>>> (bqm_and + 3.75 * i).vartype("i") == dimod.Vartype.INTEGER
True


.. note::
  It's important to remember that, for example, :code:`x = dimod.Binary('x')`
  instantiates a single-variable model with variable label ``'x'``, not a
  free-floating variable labeled ``x``. Consequently, you can add ``x`` to another
  model, say :code:`bqm = dimod.BinaryQuadraticModel('BINARY')`, by adding the two
  models, :code:`x + bqm`. This adds the variable labeled ``'x'`` in the
  single-variable BQM, ``x``, to model ``bqm``. You cannot add ``x`` to a
  model---as though it were variable ``'x'``---by doing :code:`bqm.add_variable(x)`.

Representing Constraints
========================

Many real-world problems include constraints. Typically constraints are either
equality or inequality, in the form of a left-hand side(``lhs``), right-hand-side
(``rhs``), and the "sense" (:math:`\le`, :math:`\ge`, or :math:`==`). For example,
the constraint of the rectangle problem above,

.. math::

  \textrm{s.t.} \quad 2i+2j \le P

has a ``lhs`` of :math:`2i+2j` and a ``rhs`` of a some real number (:math:`8`):

>>> print(constraint.lhs.to_polystring(), constraint.sense.value, constraint.rhs)  # doctest:+SKIP
2*i + 2*j <= 8

You can create such an equality or inequality symbolically, and it is shown
with the model:

>>> print(type(3.75 * i <= 4))
<class 'dimod.sym.Le'>
>>> 3.75 * i <= 4
QuadraticModel({'i': 3.75}, {}, 0.0, {'i': 'INTEGER'}, dtype='float64') <= 4

See the :class:`dimod.sym.Sense` class for details.

Performance
===========

*dimod*'s symbolic math is very useful for small models used for experimenting
and formulating problems. It also offers some more performant functionality; for
example, methods such as :func:`~dimod.quadratic.IntegerArray` for creating multiple
variables with NumPy arrays or :func:`~dimod.binary.quicksum` as a replacement
for the Python :func:`sum`.

See the examples of :func:`~dimod.binary.BinaryArray`, :func:`~dimod.quadratic.IntegerArray`,
and :func:`~dimod.binary.SpinArray` for usage.
