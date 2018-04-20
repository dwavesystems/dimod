.. _sampler:

=======================
Samplers and Composites
=======================

You can create your own samplers with dimod's :class:`.Sampler` abstract base class (ABC)
providing complementary methods (e.g., ‘sample_qubo’ if only ‘sample_ising’ is implemented),
consistent responses, etc.

Properties of dimod Sampler Abstract Base Classes
=================================================

The following table describes the inheritance, properties, methods/mixins of sampler
ABCs.

.. list-table::
    :header-rows: 1

    *   - ABC
        - Inherits from
        - Abstract Properties
        - Abstract Methods
        - Mixins
    *   - :class:`.Sampler`
        -
        - :attr:`~.Sampler.parameters`, :attr:`~.Sampler.properties`
        - at least one of
          :meth:`~.Sampler.sample`, :meth:`~.Sampler.sample_ising`, :meth:`~.Sampler.sample_qubo`
        - :meth:`~.Sampler.sample`, :meth:`~.Sampler.sample_ising`, :meth:`~.Sampler.sample_qubo`
    *   - :class:`.Structured`
        -
        - :attr:`~.Structured.nodelist`, :attr:`~.Structured.edgelist`
        -
        - :attr:`~.Structured.structure`, :attr:`~.Structured.adjacency`
    *   - :class:`.Composite`
        -
        - :attr:`~.Composite.children`
        -
        - :attr:`~.Composite.child`
    *   - :class:`.ComposedSampler`
        - :class:`.Sampler`, :class:`.Composite`
        - :attr:`~.Sampler.parameters`, :attr:`~.Sampler.properties`, :attr:`~.Composite.children`
        - at least one of
          :meth:`~.Sampler.sample`, :meth:`~.Sampler.sample_ising`, :meth:`~.Sampler.sample_qubo`
        - :meth:`~.Sampler.sample`, :meth:`~.Sampler.sample_ising`, :meth:`~.Sampler.sample_qubo`,
          :attr:`~.Composite.child`

The table shows, for example, that the :class:`.Sampler` class requires that you implement
the :attr:`~.Sampler.parameters` and :attr:`~.Sampler.properties` properties and at least
one sampler method; the class provides the unimplemented methods as mixins.

Creating a Sampler
==================

.. automodule:: dimod.core.sampler

.. currentmodule:: dimod
.. autoclass:: Sampler

Abstract Properties
-------------------

.. autosummary::
   :toctree: generated/

   Sampler.parameters
   Sampler.properties

Mixin Methods
-------------

.. autosummary::
   :toctree: generated/

   Sampler.sample
   Sampler.sample_ising
   Sampler.sample_qubo


Creating a Composed Sampler
===========================

.. figure:: ../_static/composing_samplers.png
    :align: center
    :name: Composing Samplers
    :scale: 70 %
    :alt: Composite Pattern.

    Composite Pattern

.. automodule:: dimod.core.composite

.. currentmodule:: dimod
.. autoclass:: ComposedSampler

.. currentmodule:: dimod
.. autoclass:: Composite

Abstract Properties
-------------------

.. autosummary::
   :toctree: generated/

   Composite.children


Mixin Properties
----------------

.. autosummary::
   :toctree: generated/

   Composite.child



Creating a Structured Sampler
=============================

.. automodule:: dimod.core.structured

.. currentmodule:: dimod
.. autoclass:: Structured

Abstract Properties
-------------------

.. autosummary::
   :toctree: generated/

   Structured.nodelist
   Structured.edgelist

Mixin Properties
----------------

.. autosummary::
   :toctree: generated/

   Structured.adjacency
   Structured.structure
