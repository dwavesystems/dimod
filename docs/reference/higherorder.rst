.. _higher_order:

===================
Higher-Order Models
===================

Sometimes it is nice to work with problems that are not restricted to quadratic
interactions.

Binary Polynomials
==================

.. automodule:: dimod.higherorder.polynomial

.. autoclass:: BinaryPolynomial

Methods
~~~~~~~

.. autosummary::
   :toctree: generated/

   BinaryPolynomial.copy
   BinaryPolynomial.energies
   BinaryPolynomial.energy
   BinaryPolynomial.from_hising
   BinaryPolynomial.from_hubo
   BinaryPolynomial.normalize
   BinaryPolynomial.relabel_variables
   BinaryPolynomial.scale
   BinaryPolynomial.to_binary
   BinaryPolynomial.to_hising
   BinaryPolynomial.to_hubo
   BinaryPolynomial.to_spin

Functions
=========

.. automodule:: dimod.higherorder.utils

.. autosummary::
   :toctree: generated/

   poly_energies
   poly_energy

Reducing to a Binary Quadratic Model
====================================

.. autosummary::
   :toctree: generated/

   make_quadratic
   make_quadratic_cqm
   reduce_binary_polynomial
