import dimod
from dimod.roof_duality import fix_variables
from dimod.response import SampleSet
import numpy as np

__all__ = ['FixedVariableComposite']


class FixedVariableComposite(dimod.ComposedSampler):

    def __init__(self, child_sampler):
        self._children = [child_sampler]

    @property
    def children(self):

        return self._children

    @property
    def parameters(self):

        return self.child.parameters.copy()

    @property
    def properties(self):

        return {'child_properties': self.child.properties.copy()}

    def sample(self, source_bqm, fixed_variables=None,
               **parameters):

        child = self.child
        if fixed_variables is None:
            fixed_variables = fix_variables(source_bqm)

        bqm = source_bqm.copy()
        bqm.fix_variables(fixed_variables)

        if len(bqm.variables) == 0:
            return _fixed_response(fixed_variables, bqm.offset, bqm.vartype)

        response = child.sample(bqm, **parameters)

        return _release_response(response, bqm, fixed_variables)


def _fixed_response(fixed_variables, offset, vartype):
    variables = fixed_variables.keys()

    samples = np.asarray([list(fixed_variables.values())])
    energy = np.asarray([offset])

    num_variables = len(fixed_variables)

    datatypes = [('sample', np.dtype(np.int8), (num_variables,)),
                 ('energy', energy.dtype),
                 ('num_occurrences', np.dtype(np.int8))]

    data = np.rec.array(np.empty(1, dtype=datatypes))

    data.sample = samples
    data.energy = energy
    data.num_occurrences = 1

    return SampleSet(data, variables,
                     {'fixed_variables': fixed_variables},
                     vartype)


def _release_response(response, bqm, fixed_variables):
    original_variables = list(bqm.variables)
    record = response.record
    samples = np.asarray(record.sample)
    energy = np.asarray(record.energy)

    num_samples, num_variables = np.shape(samples)
    num_variables += len(fixed_variables)

    b = []
    for v, val in fixed_variables.items():
        original_variables.append(v)
        b.append([val] * num_samples)
    samples = np.concatenate((samples, np.transpose(b)), axis=1)

    datatypes = [('sample', np.dtype(np.int8), (num_variables,)),
                 ('energy', energy.dtype)]

    datatypes.extend((name, record[name].dtype, record[name].shape[1:])
                     for name in record.dtype.names if
                     name not in {'sample',
                                  'energy'})

    data = np.rec.array(np.empty(num_samples, dtype=datatypes))

    data.sample = samples
    data.energy = energy
    for name in record.dtype.names:
        if name not in {'sample', 'energy'}:
            data[name] = record[name]
    response.info['fixed_variables'] = fixed_variables

    return SampleSet(data, original_variables, response.info,
                     response.vartype)
