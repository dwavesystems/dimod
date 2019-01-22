# Copyright 2018 D-Wave Systems Inc.
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
#
# ================================================================================================

"""
A composite that convert a higher order polynomial problem into a bqm by
introducing penalties before sending the bqm to its child sampler.
"""
import numpy as np

from dimod.binary_quadratic_model import BinaryQuadraticModel
from dimod.core.composite import ComposedSampler
from dimod.higherorder import make_quadratic, poly_energies, _relabeled_poly
from dimod.response import SampleSet

__all__ = ['HigherOrderComposite']


class HigherOrderComposite(ComposedSampler):
    """Reduces a HUBO to bqm by introducing penalties.

    Energies of the returned samples do not include the penalties.

   Args:
       sampler (:obj:`dimod.Sampler`):
            A dimod sampler

   Example:
       This example uses :class:`.HigherOrderComposite` to instantiate a
       composed sampler that submits a simple Ising problem to a sampler.
       The composed sampler creates a bqm from a higher order problem.

       >>> sampler = dimod.HigherOrderComposite(dimod.ExactSolver())
       >>> linear = {0: -0.5, 1: -0.3, 2: -0.8}
       >>> quadratic = {(0, 1, 2): -1.7}
       >>> response = sampler.sample_ising(linear,quadratic,penalty_strength=2,
       ...                              keep_penalty_variables=False,
       ...                              discard_unsatisfied=True)
       >>> print(response.first)  # doctest: +SKIP
       Sample(sample={0: 1, 1: 1, 2: 1}, energy=-3.3, num_occurrences=1,
              penalty_satisfaction=True)

        """

    def __init__(self, child_sampler):
        self._children = [child_sampler]

    @property
    def children(self):
        return self._children

    @property
    def parameters(self):
        param = self.child.parameters.copy()
        param['penalty_strength'] = []
        return param

    @property
    def properties(self):
        return {'child_properties': self.child.properties.copy()}

    def sample_ising(self, h, J, offset=0, penalty_strength=1.0,
                     keep_penalty_variables=False,
                     discard_unsatisfied=False, **parameters):
        """ Sample from the problem provided by h, J, offset.

        Takes in linear variables in h and quadratic and higher order
        terms in J. Introducing penalties, reduces the higher-order problem
        into a quadratic problem and send it to its child sampler.

        Args:
            h (dict): linear biases corresponding to the HUBO form

            J (dict): higher order biases corresponding to the HUBO form

            offset (float, optional): constant energy offset

            penalty_strength (float, optional): Strength of the reduction constraint.
                Insufficient strength can result in the binary quadratic model
                not having the same minimization as the polynomial.

            keep_penalty_variables (bool, optional): default is True. if False
                will remove the variables used for penalty from the samples

            discard_unsatisfied (bool, optional): default is False. If True
                will discard samples that do not satisfy the penalty conditions.

            **parameters: Parameters for the sampling method, specified by
            the child sampler.

        Returns:
            :obj:`dimod.SampleSet`
        """

        # solve the problem on the child system

        bqm = BinaryQuadraticModel.from_ising(h, {})
        bqm = make_quadratic(J, penalty_strength, bqm=bqm)
        response = self.child.sample(bqm, **parameters)

        return polymorph_response(response, h, J, bqm,
                                  offset=offset,
                                  penalty_strength=penalty_strength,
                                  keep_penalty_variables=keep_penalty_variables,
                                  discard_unsatisfied=discard_unsatisfied)


def penalty_satisfaction(response, bqm):
    """ Creates a penalty satisfaction list

    Given a sampleSet and a bqm object, will create a binary list informing
    whether the penalties introduced during degree reduction are satisfied for
    each sample in sampleSet

    Args:
        response (:obj:`.SampleSet`): Samples corresponding to provided bqm

        bqm (:obj:`.BinaryQuadraticModel`): a bqm object that contains
            its reduction info.

    Returns:
        :obj:`numpy.ndarray`: a binary array of penalty satisfaction information

    """
    record = response.record
    label_dict = response.variables.index

    if len(bqm.info['reduction']) == 0:
        return np.array([1] * len(record.sample))

    penalty_vector = np.prod([record.sample[:, label_dict[qi]] *
                              record.sample[:, label_dict[qj]]
                              == record.sample[:,
                                 label_dict[valdict['product']]]
                              for (qi, qj), valdict in
                              bqm.info['reduction'].items()], axis=0)
    return penalty_vector


def polymorph_response(response, h, J, bqm, offset=0,
                       penalty_strength=None,
                       keep_penalty_variables=True,
                       discard_unsatisfied=False):
    """ Transforms the sampleset for the higher order problem.

    Given a response of a penalized HUBO, this function creates a new sampleset
    object, taking into account penalty information and calculates the
    energies of samples for the higherorder problem.

    Args:
        response (:obj:`.SampleSet`): response for a penalized hubo.

        h (dict): linear biases corresponding to the HUBO form

        J (dict): higher order biases corresponding to the HUBO form

        bqm (:obj:`dimod.BinaryQuadraticModel`): Binary quadratic model of the
            reduced problem.

        offset (float, optional): constant energy offset

        penalty_strength (float, optional): default is None, if provided,
            will be added to the info field of the returned sampleSet object.

        keep_penalty_variables (bool, optional): default is True. if False
            will remove the variables used for penalty from the samples

        discard_unsatisfied (bool, optional): default is False. If True
            will discard samples that do not satisfy the penalty conditions.

    Returns:
        (:obj:`.SampleSet'): A sampleSet object that has additional penalty
            information. The energies of samples are calculated for the HUBO
            ignoring the penalty variables.

    """
    record = response.record
    penalty_vector = penalty_satisfaction(response, bqm)
    original_variables = bqm.variables

    if discard_unsatisfied:
        samples_to_keep = list(map(bool, list(penalty_vector)))
        penalty_vector = np.array([True] * np.sum(samples_to_keep))
    else:
        samples_to_keep = list(map(bool, [1] * len(record.sample)))

    poly = _relabeled_poly(h, J, response.variables.index)
    samples = record.sample[samples_to_keep]
    energy_vector = np.add(poly_energies(samples, poly), offset)

    if not keep_penalty_variables:
        original_variables = set(h).union(*J)
        idxs = [response.variables.index[v] for v in original_variables]
        samples = np.asarray(samples[:, idxs])

    num_samples, num_variables = np.shape(samples)

    datatypes = [('sample', np.dtype(np.int8), (num_variables,)),
                 ('energy', energy_vector.dtype),
                 ('penalty_satisfaction',
                  penalty_vector.dtype)]
    datatypes.extend((name, record[name].dtype, record[name].shape[1:])
                     for name in record.dtype.names if
                     name not in {'sample',
                                  'energy'})

    data = np.rec.array(np.empty(num_samples, dtype=datatypes))
    data.sample = samples
    data.energy = energy_vector
    for name in record.dtype.names:
        if name not in {'sample', 'energy'}:
            data[name] = record[name][samples_to_keep]

    data['penalty_satisfaction'] = penalty_vector
    response.info['reduction'] = bqm.info['reduction']
    if penalty_strength is not None:
        response.info['penalty_strength'] = penalty_strength
    return SampleSet(data, original_variables, response.info,
                     response.vartype)
