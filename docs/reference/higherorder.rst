.. _higherorder:

=============================
Solving Higher-order Problems
=============================

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

Reducing to a Binary Quadratic Model
====================================

.. automodule:: dimod.higherorder.utils

.. autosummary::
   :toctree: generated/

   make_quadratic


HigherOrderComposite
====================

.. automodule:: dimod.reference.composites.higherordercomposites

.. currentmodule:: dimod.reference.composites.higherordercomposites

.. autoclass:: HigherOrderComposite

Properties
~~~~~~~~~~

.. autosummary::
   :toctree: generated/

   HigherOrderComposite.child
   HigherOrderComposite.children
   HigherOrderComposite.parameters
   HigherOrderComposite.properties

Methods
~~~~~~~

.. autosummary::
   :toctree: generated/

   HigherOrderComposite.sample_poly
   HigherOrderComposite.sample_hising
   HigherOrderComposite.sample_hubo

PolyScaleComposite
==================

.. autoclass:: PolyScaleComposite

Properties
~~~~~~~~~~

.. autosummary::
   :toctree: generated/

   PolyScaleComposite.child
   PolyScaleComposite.children
   PolyScaleComposite.parameters
   PolyScaleComposite.properties

Methods
~~~~~~~

.. autosummary::
   :toctree: generated/

   PolyScaleComposite.sample_poly
   PolyScaleComposite.sample_hising
   PolyScaleComposite.sample_hubo

PolyTruncateComposite
=====================

.. autoclass:: PolyTruncateComposite

Properties
~~~~~~~~~~

.. autosummary::
   :toctree: generated/

   PolyTruncateComposite.child
   PolyTruncateComposite.children
   PolyTruncateComposite.parameters
   PolyTruncateComposite.properties

Methods
~~~~~~~

.. autosummary::
   :toctree: generated/

   PolyTruncateComposite.sample_poly
   PolyTruncateComposite.sample_hising
   PolyTruncateComposite.sample_hubo
