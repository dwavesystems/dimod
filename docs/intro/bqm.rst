.. _intro_bqm:

=======================
Binary Quadratic Models
=======================

dimod provides a :term:`binary quadratic model` (BQM) class that encodes
:term:`Ising` and quadratic unconstrained binary optimization (\ :term:`QUBO`\ )
models used by samplers such as the D-Wave system.

For an introduction to BQMs, see :std:doc:`Binary Quadratic Models <oceandocs:concepts/bqm>`.

dimod can represent BQMs with a few classes suited to different needs for
performance and ease of use. Beginners might start with the simple Python
:ref:`AdjDictBQM <adjdictbqm_dimod>` class (:class:`dimod.BinaryQuadraticModel`),
and later switch to a higher-performing class with a C++ implementation.

For descriptions of all supported BQM representations, see :ref:`bqm`.

Generating BQMs
===============

For instance, given
a BQM,

.. math::

   E(x) = .5 x_0 - 3 x_1 - x_0 x_1 + x_0 x_2 + 2 x_0 x_3 + x_2 x_3
