"""todo

"""
import inspect
from functools import wraps

from six import iteritems

from dimod.compatibility23 import getargspec
from dimod.exceptions import BinaryQuadraticModelStructureError
from dimod.vartypes import Vartype

__all__ = ['bqm_index_labels', 'bqm_structured', 'mixinmethod', 'vartype_argument']


def bqm_index_labels(f):
    """todo
    """

    @wraps(f)
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

    @wraps(f)
    def new_f(sampler, bqm, **kwargs):
        try:
            structure = sampler.structure
            adjacency = structure.adjacency
        except AttributeError:
            if isinstance(sampler, Structured):
                raise RuntimeError("something is wrong with the structured sampler")
            else:
                raise TypeError("sampler does not have a structure property")

        if not all(v in adjacency for v in bqm.linear):
            # todo: better error message
            raise BinaryQuadraticModelStructureError("given bqm does not match the sampler's structure")
        if not all(u in adjacency[v] for u, v in bqm.quadratic):
            # todo: better error message
            raise BinaryQuadraticModelStructureError("given bqm does not match the sampler's structure")

        return f(sampler, bqm, **kwargs)

    return new_f


def vartype_argument(*arg_names):
    """Ensures the wrapped function receives valid vartype argument(s). One
    or more argument names can be specified (as a list of string arguments).

    Args:
        *arg_names (list[str], argument names, optional, default='vartype'):
            The names of the constrained arguments in function decorated.

    Returns:
        Function decorator.

    Examples:
        >>> @dimod.vartype_argument()
        ... def f(x, vartype):
        ...     print(vartype)
        ...
        >>> f(1, 'SPIN')
        Vartype.SPIN
        >>> f(1, vartype='SPIN')
        Vartype.SPIN

        >>> @dimod.vartype_argument('y')
        ... def f(x, y):
        ...     print(y)
        ...
        >>> f(1, 'SPIN')
        Vartype.SPIN
        >>> f(1, y='SPIN')
        Vartype.SPIN

        >>> @dimod.vartype_argument('z')
        ... def f(x, **kwargs):
        ...     print(kwargs['z'])
        ...
        >>> f(1, z='SPIN')
        Vartype.SPIN

    Note:
        The function decorated can explicitly list (name) vartype arguments
        constrained by :func:`vartype_argument`, or it can use a keyword arguments `dict`.
    """
    # by default, constrain only one argument, the 'vartype`
    if not arg_names:
        arg_names = ['vartype']

    def _vartype_arg(f):
        argspec = getargspec(f)

        def _enforce_single_arg(name, args, kwargs):
            try:
                vartype = kwargs[name]
            except KeyError:
                raise TypeError('vartype argument missing')

            if isinstance(vartype, Vartype):
                return

            try:
                if isinstance(vartype, str):
                    vartype = Vartype[vartype]
                else:
                    vartype = Vartype(vartype)

            except (ValueError, KeyError):
                raise TypeError(("expected input vartype to be one of: "
                                 "Vartype.SPIN, 'SPIN', {-1, 1}, "
                                 "Vartype.BINARY, 'BINARY', or {0, 1}."))

            kwargs[name] = vartype

        @wraps(f)
        def new_f(*args, **kwargs):
            # bound actual f arguments (including defaults) to f argument names
            # (note: if call arguments don't match actual function signature,
            # we'll fail here with the standard `TypeError`)
            bound_args = inspect.getcallargs(f, *args, **kwargs)

            # `getcallargs` doesn't merge additional positional/keyword arguments,
            # so do it manually
            final_args = list(bound_args.pop(argspec.varargs, ()))
            final_kwargs = bound_args.pop(argspec.keywords, {})

            final_kwargs.update(bound_args)
            for name in arg_names:
                _enforce_single_arg(name, final_args, final_kwargs)

            return f(*final_args, **final_kwargs)

        return new_f

    return _vartype_arg
