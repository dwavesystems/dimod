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
# =============================================================================
"""Decorators can be imported from the :mod:`dimod.decorators` namespace. For
example:

>>> from dimod.decorators import vartype_argument

"""

import inspect
import itertools

try:
    import collections.abc as abc
except ImportError:
    import collections as abc

from functools import wraps
from numbers import Integral

from six import iteritems, integer_types

from dimod.compatibility23 import getargspec
from dimod.core.structured import Structured
from dimod.exceptions import BinaryQuadraticModelStructureError, WriteableError
from dimod.vartypes import as_vartype


def bqm_index_labels(f):
    """Decorator to convert a BQM to index-labels and relabel the sample set
    output.

    Designed to be applied to :meth:`.Sampler.sample`. Expects the wrapped
    function or method to accept a :obj:`.BinaryQuadraticModel` as the second
    input and to return a :obj:`.SampleSet`.

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


def bqm_index_labelled_input(var_labels_arg_name, samples_arg_names):
    """Returns a decorator that ensures BQM variable labeling and
    specified sample_like inputs are index labeled and consistent.

    Args:
        var_labels_arg_name (str):
            Expected name of the argument used to pass in an
            index labeling for the binary quadratic model (BQM).

        samples_arg_names (list[str]):
            Expected names of sample_like inputs that should be
            indexed by the labels passed to the `var_labels_arg_name`
            argument. 'samples_like' is an extension of NumPy's
            array_like_. See :func:`.as_samples`.

    Returns:
        Function decorator.

    .. _array_like: https://docs.scipy.org/doc/numpy/user/basics.creation.html
    """

    def index_label_decorator(f):
        @wraps(f)
        def _index_label(sampler, bqm, **kwargs):
            if not hasattr(bqm, 'linear'):
                raise TypeError('expected input to be a BinaryQuadraticModel')
            linear = bqm.linear

            var_labels = kwargs.get(var_labels_arg_name, None)
            has_samples_input = any(kwargs.get(arg_name, None) is not None
                                    for arg_name in samples_arg_names)

            if var_labels is None:
                # if already index-labelled, just continue
                if all(v in linear for v in range(len(bqm))):
                    return f(sampler, bqm, **kwargs)

                if has_samples_input:
                    err_str = ("Argument `{}` must be provided if any of the"
                               " samples arguments {} are provided and the "
                               "bqm is not already index-labelled".format(
                                   var_labels_arg_name,
                                   samples_arg_names))
                    raise ValueError(err_str)

                try:
                    inverse_mapping = dict(enumerate(sorted(linear)))
                except TypeError:
                    # in python3 unlike types cannot be sorted
                    inverse_mapping = dict(enumerate(linear))
                var_labels = {v: i for i, v in iteritems(inverse_mapping)}

            else:
                inverse_mapping = {i: v for v, i in iteritems(var_labels)}

            response = f(sampler,
                         bqm.relabel_variables(var_labels, inplace=False),
                         **kwargs)

            # unapply the relabeling
            return response.relabel_variables(inverse_mapping, inplace=True)

        return _index_label

    return index_label_decorator


def bqm_structured(f):
    """Decorator to raise an error if the given BQM does not match the sampler's
    structure.

    Designed to be applied to :meth:`.Sampler.sample`. Expects the wrapped
    function or method to accept a :obj:`.BinaryQuadraticModel` as the second
    input and for the :class:`.Sampler` to also be :class:`.Structured`.
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
    """Ensures the wrapped function receives valid vartype argument(s).

    One or more argument names can be specified as a list of string arguments.

    Args:
        *arg_names (list[str], argument names, optional, default='vartype'):
            Names of the constrained arguments in decorated function.

    Returns:
        Function decorator.

    Examples:

        >>> from dimod.decorators import vartype_argument

        >>> @vartype_argument()
        ... def f(x, vartype):
        ...     print(vartype)
        ...
        >>> f(1, 'SPIN')
        Vartype.SPIN
        >>> f(1, vartype='SPIN')
        Vartype.SPIN

        >>> @vartype_argument('y')
        ... def f(x, y):
        ...     print(y)
        ...
        >>> f(1, 'SPIN')
        Vartype.SPIN
        >>> f(1, y='SPIN')
        Vartype.SPIN

        >>> @vartype_argument('z')
        ... def f(x, **kwargs):
        ...     print(kwargs['z'])
        ...
        >>> f(1, z='SPIN')
        Vartype.SPIN

    Note:
        The decorated function can explicitly list (name) vartype arguments
        constrained by :func:`vartype_argument` or it can use a keyword
        arguments `dict`.

    See also:
        :func:`~dimod.as_vartype`

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

            kwargs[name] = as_vartype(vartype)

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


def _is_integer(a):
    if isinstance(a, integer_types):
        return True
    if hasattr(a, "is_integer") and a.is_integer():
        return True
    return False


# we would like to do graph_argument(*arg_names, allow_None=False), but python2...
def graph_argument(*arg_names, **options):
    """Decorator to coerce given graph arguments into a consistent form.

    The wrapped function accepts either an integer n, interpreted as a
    complete graph of size n, or a nodes/edges pair, or a NetworkX graph. The
    argument is converted into a nodes/edges 2-tuple.

    Args:
        *arg_names (optional, default='G'):
            Names of the arguments for input graphs.

        allow_None (bool, optional, default=False):
            If True, None can be passed through as an input graph.

    """

    # by default, constrain only one argument, the 'G`
    if not arg_names:
        arg_names = ['G']

    # we only allow one option allow_None
    allow_None = options.pop("allow_None", False)
    if options:
        # to keep it consistent with python3
        # behaviour like graph_argument(*arg_names, allow_None=False)
        key, _ = options.popitem()
        msg = "graph_argument() for an unexpected keyword argument '{}'".format(key)
        raise TypeError(msg)

    def _graph_arg(f):
        argspec = getargspec(f)

        def _enforce_single_arg(name, args, kwargs):
            try:
                G = kwargs[name]
            except KeyError:
                raise TypeError('Graph argument missing')

            if hasattr(G, 'edges') and hasattr(G, 'nodes'):
                # networkx or perhaps a named tuple
                kwargs[name] = (list(G.nodes), list(G.edges))

            elif _is_integer(G):
                # an integer, cast to a complete graph
                kwargs[name] = (list(range(G)), list(itertools.combinations(range(G), 2)))

            elif isinstance(G, abc.Sequence) and len(G) == 2:
                # is a pair nodes/edges
                if isinstance(G[0], integer_types):
                    # if nodes is an int
                    kwargs[name] = (list(range(G[0])), G[1])

            elif allow_None and G is None:
                # allow None to be passed through
                return G

            else:
                raise ValueError('Unexpected graph input form')

            return

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

    return _graph_arg


def lockable_method(f):
    """Method decorator for objects with an is_writeable flag.

    If wrapped method is called, and the associated object's `is_writeable`
    attribute is set to True, a :exc:`.exceptions.WriteableError` is raised.

    """
    @wraps(f)
    def _check_writeable(obj, *args, **kwds):
        if not obj.is_writeable:
            raise WriteableError
        return f(obj, *args, **kwds)
    return _check_writeable
