.. _intro_dimodqm:

===================================
Non-Binary and Non-Quadratic Models
===================================

dimod also supports :ref:`discrete quadratic models (DQMs) <dqm>` and provides some
:ref:`higher_order_composites` and functionality such as reducing higher-order
polynomials to BQMs.

Discrete Quadratic Models
=========================

For an introduction to DQMs, see :std:doc:`Discrete Quadratic Models <oceandocs:concepts/dqm>`.

See examples of using `Leap <https://cloud.dwavesys.com/leap>`_ hybrid DQM
solvers in the `dwave-examples GitHub repository <https://github.com/dwave-examples>`_.

Higher-Order Models
===================

This example uses dimod's :class:`~dimod.reference.samplers.ExactSolver` reference
sampler on a higher-order unconstrained binary optimization (HUBO) model.

>>> import dimod
>>> poly = dimod.BinaryPolynomial.from_hubo({('a', 'a'): -1,
...                                          ('a', 'b'): -0.5,
...                                          ('a', 'b', 'c'): -2})
>>> sampler = dimod.HigherOrderComposite(dimod.ExactSolver())
>>> sampleset = sampler.sample_poly(poly)
>>> print(sampleset.first.sample["a"])
1
