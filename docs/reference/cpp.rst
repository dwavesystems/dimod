.. _cppdocs_dimod:

=======
C++ API
=======

BinaryQuadraticModel
--------------------

.. doxygenclass:: dimod::BinaryQuadraticModel
    :members:
    :project: dimod

Vartype
-------

.. doxygenenum:: dimod::Vartype
    :project: dimod

.. dev note: I'd like to add vartype_limits here, but I can't figure out how to
   get doxygen to play nicely with the partial specialization. Making as TODO
   for now

Base Classes
------------

Neighborhood
~~~~~~~~~~~~

.. doxygenclass:: dimod::Neighborhood
    :members:
    :project: dimod

QuadraticModelBase
~~~~~~~~~~~~~~~~~~

.. doxygenclass:: dimod::QuadraticModelBase
    :members:
    :project: dimod
