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

Connected Components Composite
------------------------------

.. automodule:: dimod.reference.composites.connectedcomponent

Class
~~~~~

.. autoclass:: ConnectedComponentsComposite

Properties
~~~~~~~~~~

.. autosummary::
   :toctree: generated/

   ConnectedComponentsComposite.child
   ConnectedComponentsComposite.children
   ConnectedComponentsComposite.parameters
   ConnectedComponentsComposite.properties

Methods
~~~~~~~

.. autosummary::
   :toctree: generated/

   ConnectedComponentsComposite.sample
   ConnectedComponentsComposite.sample_ising
   ConnectedComponentsComposite.sample_qubo

Clip Composite
------------------------

.. automodule:: dimod.reference.composites.clipcomposite

Class
~~~~~

.. autoclass:: ClipComposite

Properties
~~~~~~~~~~

.. autosummary::
   :toctree: generated/

   ClipComposite.child
   ClipComposite.children
   ClipComposite.parameters
   ClipComposite.properties

Methods
~~~~~~~

.. autosummary::
   :toctree: generated/

   ClipComposite.sample
   ClipComposite.sample_ising
   ClipComposite.sample_qubo


Fixed Variable Composite
------------------------

.. automodule:: dimod.reference.composites.fixedvariable

Class
~~~~~

.. autoclass:: FixedVariableComposite

Properties
~~~~~~~~~~

.. autosummary::
   :toctree: generated/

   FixedVariableComposite.child
   FixedVariableComposite.children
   FixedVariableComposite.parameters
   FixedVariableComposite.properties

Methods
~~~~~~~

.. autosummary::
   :toctree: generated/

   FixedVariableComposite.sample
   FixedVariableComposite.sample_ising
   FixedVariableComposite.sample_qubo


Roof Duality Composite
----------------------

.. automodule:: dimod.reference.composites.roofduality

Class
~~~~~

.. autoclass:: RoofDualityComposite

Properties
~~~~~~~~~~

.. autosummary::
   :toctree: generated/

   RoofDualityComposite.child
   RoofDualityComposite.children
   RoofDualityComposite.parameters
   RoofDualityComposite.properties


Methods
~~~~~~~

.. autosummary::
   :toctree: generated/

   RoofDualityComposite.sample
   RoofDualityComposite.sample_ising
   RoofDualityComposite.sample_qubo


Scale Composite
---------------

.. automodule:: dimod.reference.composites.scalecomposite

Class
~~~~~

.. autoclass:: ScaleComposite

Properties
~~~~~~~~~~

.. autosummary::
   :toctree: generated/

   ScaleComposite.child
   ScaleComposite.children
   ScaleComposite.parameters
   ScaleComposite.properties

Methods
~~~~~~~

.. autosummary::
   :toctree: generated/

   ScaleComposite.sample
   ScaleComposite.sample_ising
   ScaleComposite.sample_qubo



Spin Reversal Transform Composite
---------------------------------

.. automodule:: dimod.reference.composites.spin_transform

Class
~~~~~

.. autoclass:: SpinReversalTransformComposite

Properties
~~~~~~~~~~

.. autosummary::
   :toctree: generated/

   SpinReversalTransformComposite.child
   SpinReversalTransformComposite.children
   SpinReversalTransformComposite.parameters
   SpinReversalTransformComposite.properties

Methods
~~~~~~~

.. autosummary::
   :toctree: generated/

   SpinReversalTransformComposite.sample
   SpinReversalTransformComposite.sample_ising
   SpinReversalTransformComposite.sample_qubo

Structured Composite
--------------------

.. automodule:: dimod.reference.composites.structure

Class
~~~~~

.. autoclass:: StructureComposite

Properties
~~~~~~~~~~

.. autosummary::
   :toctree: generated/

   StructureComposite.child
   StructureComposite.children
   StructureComposite.parameters
   StructureComposite.properties

Methods
~~~~~~~

.. autosummary::
   :toctree: generated/

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
   :toctree: generated/

   TrackingComposite.input
   TrackingComposite.inputs
   TrackingComposite.output
   TrackingComposite.outputs
   TrackingComposite.parameters
   TrackingComposite.properties


Methods
~~~~~~~

.. autosummary::
   :toctree: generated/

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
   :toctree: generated/

   TruncateComposite.child
   TruncateComposite.children
   TruncateComposite.parameters
   TruncateComposite.properties


Methods
~~~~~~~

.. autosummary::
   :toctree: generated/

   TruncateComposite.sample
   TruncateComposite.sample_ising
   TruncateComposite.sample_qubo
