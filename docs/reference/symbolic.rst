
.. _symbolic_math:

=============
Symbolic Math
=============

You can construct a model, for example a constrained quadratic model (CQM), from
symbolic math, which is especially useful for learning and testing with small
problems.

For an introduction to dimod's symbolic math, see :ref:`intro_symbolic_math`.

.. currentmodule:: dimod.sym

Class
=====

.. autoclass:: Sense

.. _symbolic_math_generators:

Model Generators
================

Generators for single-variable models.

Binary Quadratic Models
-----------------------

.. currentmodule:: dimod.binary

.. autosummary::
   :toctree: generated/

   Binary
   Binaries
   BinaryArray
   Spin
   Spins
   SpinArray


Quadratic Models
----------------

.. currentmodule:: dimod.quadratic

.. autosummary::
   :toctree: generated/

   Integer
   Integers
   IntegerArray
