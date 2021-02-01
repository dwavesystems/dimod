.. _adjmapbqm_dimod:

dimod.AdjMapBQM
===============

.. currentmodule:: dimod

.. autoclass:: AdjMapBQM

Attributes
----------

.. autosummary::
   :toctree: ../generated/

   ~AdjMapBQM.dtype
   ~AdjMapBQM.itype
   ~AdjMapBQM.ntype
   ~AdjMapBQM.num_interactions
   ~AdjMapBQM.num_variables
   ~AdjMapBQM.offset
   ~AdjMapBQM.shape
   ~AdjMapBQM.variables
   ~AdjMapBQM.vartype

Views
-----

.. autosummary::
   :toctree: ../generated/

   ~AdjMapBQM.adj
   ~AdjMapBQM.linear
   ~AdjMapBQM.quadratic
   ~AdjMapBQM.binary
   ~AdjMapBQM.spin

Methods
-------

.. autosummary::
   :toctree: ../generated/

   ~AdjMapBQM.add_offset
   ~AdjMapBQM.add_interaction
   ~AdjMapBQM.add_interactions_from
   ~AdjMapBQM.add_linear_equality_constraint
   ~AdjMapBQM.add_variable
   ~AdjMapBQM.add_variables_from
   ~AdjMapBQM.change_vartype
   ~AdjMapBQM.contract_variables
   ~AdjMapBQM.copy
   ~AdjMapBQM.degree
   ~AdjMapBQM.degrees
   ~AdjMapBQM.empty
   ~AdjMapBQM.energies
   ~AdjMapBQM.energy
   ~AdjMapBQM.fix_variables
   ~AdjMapBQM.flip_variable
   ~AdjMapBQM.from_coo
   ~AdjMapBQM.from_file
   ~AdjMapBQM.from_ising
   ~AdjMapBQM.from_networkx_graph
   ~AdjMapBQM.from_numpy_matrix
   ~AdjMapBQM.from_numpy_vectors
   ~AdjMapBQM.from_qubo
   ~AdjMapBQM.get_linear
   ~AdjMapBQM.get_quadratic
   ~AdjMapBQM.has_variable
   ~AdjMapBQM.iter_interactions
   ~AdjMapBQM.iter_linear
   ~AdjMapBQM.iter_neighbors
   ~AdjMapBQM.iter_quadratic
   ~AdjMapBQM.iter_variables
   ~AdjMapBQM.normalize
   ~AdjMapBQM.relabel_variables
   ~AdjMapBQM.relabel_variables_as_integers
   ~AdjMapBQM.remove_interaction
   ~AdjMapBQM.remove_interactions_from
   ~AdjMapBQM.remove_offset
   ~AdjMapBQM.remove_variable
   ~AdjMapBQM.remove_variables_from
   ~AdjMapBQM.scale
   ~AdjMapBQM.set_linear
   ~AdjMapBQM.set_quadratic
   ~AdjMapBQM.shapeable
   ~AdjMapBQM.to_coo
   ~AdjMapBQM.to_file
   ~AdjMapBQM.to_ising
   ~AdjMapBQM.to_networkx_graph
   ~AdjMapBQM.to_numpy_matrix
   ~AdjMapBQM.to_numpy_vectors
   ~AdjMapBQM.to_qubo
   ~AdjMapBQM.update
