.. _quadratic_composites:

==========
Composites
==========

The `dimod` package includes several example composed samplers:

.. currentmodule:: dimod.reference.composites

.. contents::
    :local:
    :depth: 1

The :std:doc:`dwave-system <oceandocs:docs_system/sdk_index>` package provides 
additional :std:doc:`composites for D-Wave systems <oceandocs:docs_system/reference/composites>`
such as those used for :term:`minor-embedding`.

Structure Composite
-------------------

.. automodule:: dimod.reference.composites.structure

Class
~~~~~

.. autoclass:: StructureComposite

Properties
~~~~~~~~~~

.. autosummary::
   :toctree: ../generated/

   StructureComposite.child
   StructureComposite.children
   StructureComposite.parameters
   StructureComposite.properties

Methods
~~~~~~~

.. autosummary::
   :toctree: ../generated/

   StructureComposite.sample
   StructureComposite.sample_ising
   StructureComposite.sample_qubo


Tracking Composite
------------------

.. automodule:: dimod.reference.composites.tracking

Class
~~~~~

.. autoclass:: TrackingComposite

Properties
~~~~~~~~~~

.. autosummary::
   :toctree: ../generated/

   TrackingComposite.input
   TrackingComposite.inputs
   TrackingComposite.output
   TrackingComposite.outputs
   TrackingComposite.parameters
   TrackingComposite.properties


Methods
~~~~~~~

.. autosummary::
   :toctree: ../generated/

   TrackingComposite.clear
   TrackingComposite.sample
   TrackingComposite.sample_ising
   TrackingComposite.sample_qubo


Truncate Composite
------------------

.. automodule:: dimod.reference.composites.truncatecomposite

Class
~~~~~

.. autoclass:: TruncateComposite

Properties
~~~~~~~~~~

.. autosummary::
   :toctree: ../generated/

   TruncateComposite.child
   TruncateComposite.children
   TruncateComposite.parameters
   TruncateComposite.properties


Methods
~~~~~~~

.. autosummary::
   :toctree: ../generated/

   TruncateComposite.sample
   TruncateComposite.sample_ising
   TruncateComposite.sample_qubo
