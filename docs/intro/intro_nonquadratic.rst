.. _intro_nonquadratic:

===================
Higher-Order Models
===================

dimod provides some :ref:`higher_order_composites` and functionality
such as reducing higher-order polynomials to BQMs.

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
