.. _intro_cqm:

=============================
Quadratic Models: Constrained
=============================

Many real-world problems include constraints. For example, a routing problem
might limit the number of airplanes on the ground at an airport and a scheduling
problem might require a minimum interval between shifts.

When using unconstrained samplers to handle problems with constraints, you
typically formulate the constraints as penalties. Constrained models such as
:class:`ConstrainedQuadraticModel` can support constraints by encoding both an
objective and its set of constraints, as models or in symbolic form.

Supported Models
================

dimod provides a :term:`constrained quadratic model` (CQM) class that encodes
a quadratic objective and one or more quadratic equality and inequality constraints.

For an introduction to CQMs, see
:std:doc:`Constrained Quadratic Models <oceandocs:concepts/cqm>`.

For descriptions of the CQM class and its methods, see :ref:`cqm`.

Model Generation
================

dimod provides a variety of model generators.

Example: dimod CQM Generator
----------------------------

>>>

Typically you construct a model when reformulating your problem, using such
techniques as those presented in
:std:doc:`D-Wave's system documentation <oceandocs:doc_handbook>`.

Example: CQM Construction
-------------------------

The

As mentioned above, for learning and testing with small BQMs, dimod's symbolic
construction of CQMs is convenient:

>>> x = BINARY('x')
>>> y = INTEGER('y')
>>> cqm = dimod.CQM()
>>> cqm.add_()

Especially for very large BQMs, you might read the data from a file using methods,
such as :meth:`~dimod.bqm.adjvectorbqm.AdjVectorBQM.from_file` or others,
described in the documentation of each class.

Model Attributes
================

dimod's model objects provide access to a number of attributes and views. See the
documentation for a particular type of model class.

Example:
----------------

>>> dict_bqm.shape
(4, 3)


Model Methods
=============

BQMs support a large number of methods, many common, some particular to a class,
described under the documentation for :ref:`each class <bqm>`, to enable you to
build and manipulate BQMs.

>>> map_bqm.num_interactions
21
>>> map_bqm.remove_interaction(5, 6)
>>> map_bqm.num_interactions
20
