.. _dimod_ess:

=========
Effective Sample Size
=========

The effective sample size (ESS; a formally defined statistic) quantifies the number of
effectively independent samples drawn from an autocorrelated sampler. For example, samples from
a Markov chain sampler such as Metropolis-Hastings. For another example, samples from an annealing
quantum computer modelled as a Markov chain (due to low-frequency noise). The quantification of
this notion of independence hinges on defining a test function mapping binary strings to real numbers.
Two such examples of test functions are magnetization and energy.

The univariate estimator implemented here is the (multivariate) estimator defined in the first
equation at the top of page 14 in
`Revisiting the Gelman-Rubin Diagnostic <https://arxiv.org/abs/1812.09384>`_.

For an introduction to effective sample size, see the
`Stan Manual <https://mc-stan.org/docs/2_21/reference-manual/effective-sample-size-section.html>`_
or this
`blog post <https://andrewcharlesjones.github.io/journal/21-effective-sample-size.html>_.

.. currentmodule:: dimod.ess

.. autosummary::
   :toctree: generated/

   compute_ess
   compute_ess_sampleset
