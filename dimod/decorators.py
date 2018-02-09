"""todo

"""
from dimod.compatibility23 import iteritems
from dimod.exceptions import BinaryQuadraticModelStructureError
from dimod.vartypes import Vartype


def bqm_index_labels(f):
    """todo

    """
    def _index_label(sampler, bqm, **kwargs):
        if not hasattr(bqm, 'linear'):
            raise TypeError('expected input to be a BinaryQuadraticModel')
        linear = bqm.linear

        # if already index-labelled, just continue
        if all(v in linear for v in range(len(bqm))):
            return f(sampler, bqm, **kwargs)

        try:
            inverse_mapping = dict(enumerate(sorted(linear)))
        except TypeError:
            # in python3 unlike types cannot be sorted
            inverse_mapping = dict(enumerate(linear))
        mapping = {v: i for i, v in iteritems(inverse_mapping)}

        response = f(sampler, bqm.relabel_variables(mapping, inplace=False), **kwargs)

        # unapply the relabeling
        return response.relabel_variables(inverse_mapping, inplace=True)

    return _index_label


def bqm_structured(f):
    """todo

    makes sure bqm has the appropriate structure
    """
    def new_f(sampler, bqm, **kwargs):
        try:
            structure = sampler.structure
        except AttributeError:
            if isinstance(sampler, Structured):
                raise RuntimeError("something is wrong with the structured sampler")
            else:
                raise TypeError("sampler does not have a structure property")

        if not all(v in structure.adjacency for v in bqm.linear):
            # todo: better error message
            raise BinaryQuadraticModelStructureError("given bqm does not match the sampler's structure")
        if not all(u in structure.adjacency[v] for u, v in bqm.quadratic):
            # todo: better error message
            raise BinaryQuadraticModelStructureError("given bqm does not match the sampler's structure")

        return sampler.sample(bqm, **kwargs)
    return new_f


def vartype_argument(arg_idx):
    """todo"""
    def _vartype_arg(f):
        def new_f(*args, **kwargs):

            try:
                vartype = args[arg_idx]
            except IndexError:
                vartype = kwargs['vartype']

            if isinstance(vartype, Vartype):
                # we don't need to do anything
                return f(*args, **kwargs)

            try:
                if isinstance(vartype, str):
                    vartype = Vartype[vartype]
                else:
                    vartype = Vartype(vartype)

            except (ValueError, KeyError):
                raise TypeError(("expected input vartype to be one of: "
                                 "Vartype.SPIN, 'SPIN', {-1, 1}, "
                                 "Vartype.BINARY, 'BINARY', or {0, 1}."))

            new_args = list(args)
            new_args[arg_idx] = vartype

            return f(*new_args, **kwargs)
        new_f.__name__ = f.__name__
        return new_f
    return _vartype_arg
