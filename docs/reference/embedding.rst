.. _embedding:

=========
Embedding
=========

Provides functions that map binay quadratic models and samples between a source graph and a target graph.

.. glossary::

    model
        A collection of variables with associated linear and
        quadratic biases. Sometimes referred to in other projects as a **problem**.
        In this project all models are expected to be spin-valued - that is the
        variables in the model can be -1 or 1.

    graph
        A collection of nodes and edges. A graph can be derived
        from a model; a node for each variable and an edge for each pair
        of variables with a non-zero quadratic bias.

    source
        The model or induced graph that we wish to embed. Sometimes
        referred to in other projects as the **logical** graph/model.

    target
        Embedding attempts to create a target model from a target
        graph. The process of embedding takes a source model, derives the source
        graph, maps the source graph to the target graph, then derives the target
        model. Sometimes referred to in other projects at the **embedded** graph/model.

    chain
        A collection of nodes or variables in the target graph/model
        that we want to act like a single node/variable.

    chain strength
        The magnitude of the negative quadratic bias applied
        between variables within a chain.

Functions
=========

.. currentmodule:: dimod
.. autosummary::
   :toctree: generated/

   embed_bqm
   embed_ising
   embed_qubo
   iter_unembed
   unembed_response
   chain_break_frequency

Chain Break Resolution
======================

.. automodule:: dimod.embedding.chain_breaks

.. currentmodule:: dimod.embedding

Functions
---------

.. autosummary::
   :toctree: generated/

   discard
   majority_vote
   weighted_random

Callable Objects
----------------

.. autosummary::
   :toctree: minimize_energy

   MinimizeEnergy