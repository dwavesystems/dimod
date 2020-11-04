.. _intro_dimod:

============
Introduction
============

.. include:: ..\README.rst
  :start-after: index-start-marker1
  :end-before: index-end-marker1

For explanations of the terminology, see the
:std:doc:`Ocean glossary <oceandocs:concepts/glossary>`.

The following sections give an orientation to dimod with usage examples:

.. toctree::
  :maxdepth: 1

  intro_bqm
  intro_samplers
  intro_samples

..
      It provides useful functionality for working with these models and samplers;
      for example :ref:`generators_dimod` to build BQMs and :ref:`utilities_dimod` for calculating the energy of a
      sample or serializing dimod objects.

      It also provides an :ref:`api` for constructing new samplers and composed samplers
      tailored for your problem.

      Additionally, it provides some :ref:`higher_order_composites` and functionality
      such as reducing higher-order polynomials to BQMs.
