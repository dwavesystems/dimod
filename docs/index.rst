..  -*- coding: utf-8 -*-

.. _contents:

=====
dimod
=====

.. include:: ../README.rst
  :start-after: index-start-marker1
  :end-before: index-end-marker1

Example Usage
-------------

The QUBO form, :math:`\text{E}(a_i, b_{i,j}; q_i) = -q_1 -q_2 + 2q_1 q_2`,
is related to the Ising form, :math:`\text{E}(h_i, j_{i,j}; s_i) = \frac{1}{2}(s_1s_2-1)`,
via the simple manipulation :math:`s_i=2q_i-1`.

.. include:: ../README.rst
  :start-after: index-start-marker2
  :end-before: index-end-marker2



Documentation
-------------

.. only:: html

  :Release: |version|
  :Date: |today|

.. note:: This documentation is for the latest version of
   `dimod <https://github.com/dwavesystems/dimod>`_.
   Documentation for the version currently installed by
   `dwave-ocean-sdk <https://github.com/dwavesystems/dwave-ocean-sdk>`_
   is here: :std:doc:`dimod <oceandocs:docs_dimod/sdk_index>`.

.. sdk-start-marker

.. toctree::
  :maxdepth: 1

  introduction
  reference/index
  bibliography

.. sdk-end-marker

.. toctree::
  :caption: Code
  :maxdepth: 1

  Source <https://github.com/dwavesystems/dimod>
  installation
  license

.. toctree::
  :caption: Ocean Software
  :maxdepth: 1

  Ocean Home <https://ocean.dwavesys.com/>
  Ocean Documentation <https://docs.ocean.dwavesys.com>
  Ocean Glossary <https://docs.ocean.dwavesys.com/en/latest/glossary.html>

.. toctree::
  :caption: D-Wave
  :maxdepth: 1

  D-Wave <https://www.dwavesys.com>
  Leap <https://cloud.dwavesys.com/leap/>
  D-Wave System Documentation <https://docs.dwavesys.com/docs/latest/index.html>

Indices and tables
------------------

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
* :ref:`glossary`
