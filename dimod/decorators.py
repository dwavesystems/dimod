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

import collections.abc as abc
import inspect
import itertools
import warnings

from functools import wraps
from numbers import Integral

from dimod.exceptions import BinaryQuadraticModelStructureError, WriteableError
from dimod.utilities import new_variable_label
from dimod.vartypes import as_vartype

__all__ = ['nonblocking_sample_method',
           'bqm_index_labels',
           'bqm_index_labelled_input'
           'bqm_structured',
           'vartype_argument',
           'graph_argument',
           'forwarding_method',
           ]


def nonblocking_sample_method(f):
    """Decorator to create non-blocking sample methods.

    Some samplers work asynchronously, and it is useful for composites to
    handle that case. This decorator can be used to easily construct a
    non-blocking :class:`.Sampler` or :class:`.Composite`.

    The function being decorated must return an iterator when called. This
    iterator must yield exactly two values.

    The first value can be any object, but if the object has a `done()`
    method, that method will determine the value of :meth:`.SampleSet.done()`.

    The second value must be a :class:`~dimod.SampleSet`, which will provide the
    samples to the user.

    The generator is executed until the first yield. The generator is then
    resumed when the returned sample set is resolved.

    >>> from dimod.decorators import nonblocking_sample_method
    ...
    >>> class Sampler:
    ...     @nonblocking_sample_method
    ...     def sample(self, bqm):
    ...         print("First part!")
    ...         yield
    ...         print("Second part!")
    ...         sample = {v: 1 for v in bqm.variables}
    ...         yield dimod.SampleSet.from_samples_bqm(sample, bqm)
    ...
    >>> bqm = dimod.BinaryQuadraticModel.from_ising({'a': -1}, {('a', 'b'): 1})
    >>> ss = Sampler().sample(bqm)
    First part!
    >>> ss.resolve()
    Second part!
    >>> print(ss)
       a  b energy num_oc.
    0 +1 +1    0.0       1
    ['SPIN', 1 rows, 1 samples, 2 variables]

    """
    from dimod.sampleset import SampleSet  # avoid circular import

    @wraps(f)
    def _sample(*args, **kwargs):
        iterator = f(*args, **kwargs)

        # resolve blocking part now, and make hook for the non-blocking part
        return SampleSet.from_future(next(iterator), lambda _: next(iterator))
    return _sample


def bqm_index_labels(f):
    """Decorator to convert a BQM to index-labels and relabel the sample set
    output.

    Designed to be applied to :meth:`.Sampler.sample`. Expects the wrapped
    function or method to accept a :obj:`.BinaryQuadraticModel` as the second
    input and to return a :obj:`~dimod.SampleSet`.

    """
    @wraps(f)
    def _index_label(sampler, bqm, **kwargs):
        bqm, mapping = bqm.relabel_variables_as_integers(inplace=False)
        return f(sampler, bqm, **kwargs).relabel_variables(mapping, inplace=False)
    return _index_label


def bqm_structured(f):
    """Decorator to raise an error if the given BQM does not match the sampler's
    structure.

    Designed to be applied to :meth:`.Sampler.sample`. Expects the wrapped
    function or method to accept a :obj:`.BinaryQuadraticModel` as the second
    input and for the :class:`.Sampler` to also be :class:`.Structured`.
    """
    @wraps(f)
    def structured_sample(sampler, bqm, **kwargs):
        adjacency = sampler.adjacency

        if not adjacency.keys() >= bqm.variables:
            raise BinaryQuadraticModelStructureError(
                f"given bqm has at least one variable, {(list(bqm.variables) - adjacency.keys()).pop()!r}, "
                "not supported by the structured solver")
        if any(u not in adjacency[v] for u, v, _ in bqm.iter_quadratic()):
            u, v = next((u, v) for u, v, _ in bqm.iter_quadratic() if u not in adjacency[v])
            raise BinaryQuadraticModelStructureError(
                f"given bqm contains an interaction, {(u, v)!r}, "
                "not supported by the structured solver")
        return f(sampler, bqm, **kwargs)
    return structured_sample


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
        argspec = inspect.getfullargspec(f)

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
            final_kwargs = bound_args.pop(argspec.varkw, {})

            final_kwargs.update(bound_args)
            for name in arg_names:
                _enforce_single_arg(name, final_args, final_kwargs)

            return f(*final_args, **final_kwargs)

        return new_f

    return _vartype_arg


def _is_integer(a):
    if isinstance(a, int):
        return True
    if hasattr(a, "is_integer") and a.is_integer():
        return True
    return False


# we would like to do graph_argument(*arg_names, allow_None=False), but python2...
def graph_argument(*arg_names, **options):
    """Decorator to coerce given graph arguments into a consistent form.

    The wrapped function accepts either an integer n, interpreted as a
    complete graph of size n, a nodes/edges pair, a sequence of edges, or a
    NetworkX graph. The argument is converted into a nodes/edges 2-tuple.

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
        argspec = inspect.getfullargspec(f)

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

            elif isinstance(G, abc.Sequence):
                if len(G) != 2:
                    # edgelist
                    kwargs[name] = (list(set().union(*G)), G)
                else:  # len(G) == 2
                    # need to determine if this is a nodes/edges pair or an
                    # edgelist
                    if isinstance(G[0], int):
                        # nodes are an int so definitely nodelist
                        kwargs[name] = (list(range(G[0])), G[1])
                    elif all(isinstance(e, abc.Sequence) and len(e) == 2
                             for e in G):
                        # ok, everything is a sequence and everything has length
                        # 2, so probably an edgelist. But we're dealing with
                        # only four objects so might as well check to be sure
                        nodes, edges = G
                        if all(isinstance(e, abc.Sequence) and len(e) == 2 and
                               (v in nodes for v in e) for e in edges):
                            pass  # nodes, edges
                        else:
                            # edgelist
                            kwargs[name] = (list(set().union(*G)), G)
                    else:
                        # nodes, edges
                        pass

            elif allow_None and G is None:
                # allow None to be passed through
                kwargs[name] = G

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
            final_kwargs = bound_args.pop(argspec.varkw, {})

            final_kwargs.update(bound_args)
            for name in arg_names:
                _enforce_single_arg(name, final_args, final_kwargs)

            return f(*final_args, **final_kwargs)

        return new_f

    return _graph_arg


_NOT_FOUND = object()


def forwarding_method(func):
    """Improve the performance of a forwarding method by avoiding an attribute
    lookup.

    The decorated method should return the function that it is forwarding to.
    Subsequent calls will be made directly to that function.

    Example:

        >>> import typing
        >>> import timeit
        >>> from dimod.decorators import forwarding_method
        ...
        >>> class Inner:
        ...     def func(self, a: int, b: int = 0) -> int:
        ...         "Inner.func docsting."
        ...         return a + b
        ...
        >>> class Outer:
        ...     def __init__(self):
        ...         self.inner = Inner()
        ...
        ...     def func(self, a: int, b: int = 0) -> int:
        ...         "Outer.func docsting."
        ...         return self.inner.func(a, b=b)
        ...
        ...     @forwarding_method
        ...     def fwd_func(self, a: int, b: int = 0) -> int:
        ...         "Outer.fwd_func docsting."
        ...         return self.inner.func
        ...
        >>> obj = Outer()
        >>> obj.func(2, 3)
        5
        >>> obj.fwd_func(1, 3)
        4
        >>> timeit.timeit(lambda: obj.func(10, 5))  # doctest:+SKIP
        0.275462614998105
        >>> timeit.timeit(lambda: obj.fwd_func(10, 5))  # doctest:+SKIP
        0.16692455199881806
        >>> Outer.fwd_func.__doc__
        'Outer.fwd_func docsting.'
        >>> obj.fwd_func.__doc__
        'Inner.func docsting.'

    """
    @wraps(func)
    def wrapper(obj, *args, **kwargs):
        name = func.__name__

        try:
            cache = obj.__dict__
        except AttributeError:
            raise TypeError(
                f"No '__dict__' attribute on {type(obj).__name__!r}") from None

        method = cache.get(name, _NOT_FOUND)

        if method is _NOT_FOUND:
            # the args and kwargs are ignored but they are required to not
            # raise an error
            method = func(obj, *args, **kwargs)
            try:
                cache[name] = method
            except TypeError:
                raise TypeError(
                    f"the '__dict__' attribute of {type(obj).__name__!r} "
                    "instance does not support item assignment.") from None

        return method(*args, **kwargs)

    return wrapper

def unique_variable_labels(f):
    """Decorator to assign unique labels to variables when no label is passed.

    Designed to be applied to variable methods, :meth:`.dimod.Binary`,
    :meth:`.dimod.Spin`, :meth:`.dimod.Integer`.

    """
    @wraps(f)
    def conditional_unique_label(label = None, *args, **kwargs):
        if label is None:
            qm = f(label=new_variable_label(), *args, **kwargs)
            return qm
        qm = f(label, *args, **kwargs)
        return qm

    return conditional_unique_label
