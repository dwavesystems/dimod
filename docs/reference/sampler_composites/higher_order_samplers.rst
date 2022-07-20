.. _higher_order_samplers:

=====================
Higher-Order Samplers
=====================

The `dimod` package includes the following example higher-order samplers.

.. currentmodule:: dimod.reference.samplers

Exact Polynomial Solver
-----------------------

A simple exact solver for testing and debugging code using your local CPU.

Note:
    This sampler is designed for use in testing. Because it calculates the
    energy for every possible sample, it is very slow.

Class
~~~~~

.. autoclass:: ExactPolySolver

Methods
~~~~~~~

.. autosummary::
   :toctree: ../generated/

   ExactPolySolver.sample_hising
   ExactPolySolver.sample_hubo
   ExactPolySolver.sample_poly
