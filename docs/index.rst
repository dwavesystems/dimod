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

.. toctree::
  :maxdepth: 1

  introduction
  reference/index
  installation
  license
  bibliography
  Source <https://github.com/dwavesystems/dimod>

.. toctree::
  :caption: D-Wave's Ocean Software
  :maxdepth: 1

  ocean
  contributing
  glossary

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
