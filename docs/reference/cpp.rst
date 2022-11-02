.. _cppdocs_dimod:

=======
C++ API
=======

dimod
=====

BinaryQuadraticModel
--------------------

.. doxygenclass:: dimod::BinaryQuadraticModel
    :members:
    :project: dimod

QuadraticModel
--------------

.. doxygenclass:: dimod::QuadraticModel
    :members:
    :project: dimod

Vartype
-------

.. doxygenenum:: dimod::Vartype
   :project: dimod

vartype_info
------------

.. doxygenclass:: dimod::vartype_info
    :members:
    :undoc-members:
    :project: dimod

.. Todo: vartype_limits. Getting it to look nice is possible but fiddly

dimod::abc
==========

.. doxygenclass:: dimod::abc::QuadraticModelBase
    :members:
    :protected-members:
    :project: dimod

.. Todo: dimod lp

dimod::utils
============

.. doxygenfunction:: zip_sort
   :project: dimod
