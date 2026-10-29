# Copyright 2022 D-Wave Systems Inc.
#
#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at
#
#        http://www.apache.org/licenses/LICENSE-2.0
#
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.

"""
A class for presolve algorithms.

Presolve algorithms enhance performance and solution quality by performing preprocessing
to reduce a problem’s redundant variables and constraints and to improve the
accuracy of the CQM.

The purpose of the following example is to show how to locally run the presolve
algorithms provided here prior to submitting your CQM to a solver, such as the
:class:`~dwave.system.samplers.LeapHybridCQMSampler`.

This example uses a simplified problem for illustrative purposes: a CQM with just
a single-variable objective and constraint. This is sufficient to
show how to run :class:`~dwave.preprocessing.presolve.Presolver` methods.

The CQM created below has a constraint requiring that integer variable ``j``
not exceed 5.

>>> import dimod
...
>>> j = dimod.Integer("j")
>>> cqm = dimod.ConstrainedQuadraticModel()
>>> cqm.set_objective(j)
>>> cqm.add_constraint(j <= 5, "Maximum j")
'Maximum j'

Clearly, the global optimum for this CQM occurs for the default value of the lower
bound of ``j``.

>>> cqm.lower_bound("j")
0.0

To run Ocean's presolve algorithms locally, instantiate a :class:`~dwave.preprocessing.presolve.Presolver`
on your CQM and apply a supported presolve (default is used here).

>>> from dwave.preprocessing.presolve import Presolver
...
>>> presolve = Presolver(cqm)
>>> presolve.apply()
True

You now have a preprocessed CQM you can submit to a CQM solver such as a Leap CQM solver.

>>> reduced_cqm = presolve.detach_model()
>>> print(reduced_cqm.constraints)
{}

The dimod :class:`~dimod.reference.samplers.ExactCQMSolver` test solver is
capable of solving this very simple CQM.

>>> sampleset = dimod.ExactCQMSolver().sample_cqm(reduced_cqm)
>>> print(sampleset.first)
Sample(sample={0: 0}, energy=0.0, num_occurrences=1, is_satisfied=array([], dtype=bool), is_feasible=True)

View the solution as assignments of the original CQM's variables:

>>> presolve.restore_samples(sampleset.first.sample)
(array([[0.]]), Variables(['j']))

You can also create the sample set for the original CQM:

>>> restored_sampleset = dimod.SampleSet.from_samples_cqm(presolve.restore_samples(sampleset.samples()), cqm)
>>> print(restored_sampleset)
    j energy num_oc. is_sat. is_fea.
0 0.0    0.0       1 arra...    True
1 1.0    1.0       1 arra...    True
2 2.0    2.0       1 arra...    True
3 3.0    3.0       1 arra...    True
4 4.0    4.0       1 arra...    True
5 5.0    5.0       1 arra...    True
['INTEGER', 6 rows, 6 samples, 1 variables]

"""
import warnings

import dimod

from dwave.preprocessing.presolve.cypresolve import cyPresolver, Feasibility, TechniqueFlags

__all__ = ["Feasibility", "Presolver", "TechniqueFlags"]


class Presolver(cyPresolver):
    """Presolver for constrained quadratic models.

    The model held by this class to represent the instantiating constrained
    quadratic model (CQM) is index-labeled. This is because presolve may remove,
    add, change the type of, and substitute variables. Consequently, while the
    models remain mathematically equivalent, variables of the original and reduced
    CQMs may not have a direct relationship.

    Args:
        cqm: A :class:`dimod.ConstrainedQuadraticModel`.
        move: If ``True``, the original CQM is cleared and its contents are moved
            to the presolver. This is useful for large models where memory is a concern.

    Example:

        This example reduces an implicitly fixed constraint.

        >>> import dimod
        >>> from dwave.preprocessing import Presolver

        Create a simple CQM with one variable fixed by bounds.

        >>> cqm = dimod.ConstrainedQuadraticModel()
        >>> i = dimod.Integer('i', lower_bound=-5, upper_bound=5)
        >>> j = dimod.Integer('j', lower_bound=5, upper_bound=10)
        >>> cqm.set_objective(i + j)
        >>> c0 = cqm.add_constraint(j <= 5)  # implicitly fixes 'j'

        Run presolve with default settings.

        >>> presolver = Presolver(cqm)
        >>> presolver.apply()
        True

        The model is reduced.

        >>> reduced_cqm = presolver.detach_model()
        >>> reduced_cqm.num_variables()
        1
        >>> reduced_cqm.num_constraints()
        0

    """
    # include this for the function signature
    def __init__(self, cqm: dimod.ConstrainedQuadraticModel, *, move: bool = False):
        super().__init__(cqm, move=move)

    def add_techniques(self, techniques: TechniqueFlags) -> TechniqueFlags:
        """Add presolve techniques to be used by the presolver.

        This method adds to the techniques that are already loaded.

        Args:
            techniques (:class:`TechniqueFlags`):
                The techniques to be used.

        Returns:
            :class:`TechniqueFlags`: The currently loaded presolve techniques.

        """
        return self.set_techniques(self.techniques() | techniques)

    def load_default_presolvers(self):
        """Load the default presolvers.

        .. deprecated:: 0.6.0

           This method was deprecated in dwave-preprocessing 0.6.0.
           Use :meth:`set_techniques` with :class:`TechniqueFlags.Default` instead.

        """
        warnings.warn("load_default_presolvers was deprecated in dwave-preprocessing 0.6.0, "
                      "use set_techniques(TechniqueFlags.Default) instead.",
                      DeprecationWarning, stacklevel=2)

        self.set_techniques(TechniqueFlags.Default)
