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

CQM
===

TODO
