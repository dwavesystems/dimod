"""todo

"""
from dimod.compatibility23 import iteritems
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

        response = f(sampler, bqm.relabel_variables(mapping, copy=True), **kwargs)

        # unapply the relabeling
        raise NotImplementedError
        return response

    return _index_label


def patch_sample_kwargs(f):
    """todo

    """
    def _patch_kwargs(sampler, *args, **kwargs):
        if sampler.default_sample_kwargs:
            default_kwargs = sampler.default_sample_kwargs
            default_kwargs.update(kwargs)  # overwrite with provided
            return f(sampler, *args, **default_kwargs)
        else:
            return f(sampler, *args, **kwargs)

    return _patch_kwargs


def vartype_argument(arg_idx):
    """todo"""
    def _vartype_arg(f):
        def new_f(*args, **kwargs):

            vartype = args[arg_idx]

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
