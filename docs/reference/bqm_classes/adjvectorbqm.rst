.. _adjvectorbqm_dimod:

dimod.AdjVectorBQM
==================

.. currentmodule:: dimod

.. autoclass:: AdjVectorBQM

Attributes
----------

.. autosummary::
   :toctree: ../generated/

   ~AdjVectorBQM.dtype
   ~AdjVectorBQM.itype
   ~AdjVectorBQM.ntype
   ~AdjVectorBQM.num_interactions
   ~AdjVectorBQM.num_variables
   ~AdjVectorBQM.offset
   ~AdjVectorBQM.shape
   ~AdjVectorBQM.variables
   ~AdjVectorBQM.vartype

Views
-----

.. autosummary::
   :toctree: ../generated/

   ~AdjVectorBQM.adj
   ~AdjVectorBQM.linear
   ~AdjVectorBQM.quadratic
   ~AdjVectorBQM.binary
   ~AdjVectorBQM.spin

Methods
-------

.. autosummary::
   :toctree: ../generated/

   ~AdjVectorBQM.add_offset
   ~AdjVectorBQM.add_interaction
   ~AdjVectorBQM.add_interactions_from
   ~AdjVectorBQM.add_variable
   ~AdjVectorBQM.add_variables_from
   ~AdjVectorBQM.change_vartype
   ~AdjVectorBQM.contract_variables
   ~AdjVectorBQM.copy
   ~AdjVectorBQM.degree
   ~AdjVectorBQM.degrees
   ~AdjVectorBQM.empty
   ~AdjVectorBQM.energies
   ~AdjVectorBQM.energy
   ~AdjVectorBQM.fix_variables
   ~AdjVectorBQM.flip_variable
   ~AdjVectorBQM.from_coo
   ~AdjVectorBQM.from_file
   ~AdjVectorBQM.from_ising
   ~AdjVectorBQM.from_networkx_graph
   ~AdjVectorBQM.from_numpy_matrix
   ~AdjVectorBQM.from_numpy_vectors
   ~AdjVectorBQM.from_qubo
   ~AdjVectorBQM.get_linear
   ~AdjVectorBQM.get_quadratic
   ~AdjVectorBQM.has_variable
   ~AdjVectorBQM.iter_interactions
   ~AdjVectorBQM.iter_linear
   ~AdjVectorBQM.iter_neighbors
   ~AdjVectorBQM.iter_quadratic
   ~AdjVectorBQM.iter_variables
   ~AdjVectorBQM.normalize
   ~AdjVectorBQM.relabel_variables
   ~AdjVectorBQM.relabel_variables_as_integers
   ~AdjVectorBQM.scale
   ~AdjVectorBQM.set_linear
   ~AdjVectorBQM.set_quadratic
   ~AdjVectorBQM.shapeable
   ~AdjVectorBQM.to_coo
   ~AdjVectorBQM.to_file
   ~AdjVectorBQM.to_ising
   ~AdjVectorBQM.to_networkx_graph
   ~AdjVectorBQM.to_numpy_matrix
   ~AdjVectorBQM.to_numpy_vectors
   ~AdjVectorBQM.to_qubo
   ~AdjVectorBQM.remove_interaction
   ~AdjVectorBQM.remove_interactions_from
   ~AdjVectorBQM.remove_offset
   ~AdjVectorBQM.remove_variable
   ~AdjVectorBQM.remove_variables_from
   ~AdjVectorBQM.update
