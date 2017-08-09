"""
Examples
--------
    Define a sampler that operates on QUBO problems:

    >>> class MyLinearSampler(dimod.TemplateSampler):
    ...     @dimod.decorators.qubo(1)
    ...     def sample_qubo(self, Q):
    ...         response = dimod.BinaryResponse()
    ...
    ...         sample = {}
    ...         for (u, v) in Q:
    ...             if u != v:
    ...                 continue
    ...
    ...             if Q[(u, v)] > 0:
    ...                 val = 0
    ...             else:
    ...                 val = 1
    ...
    ...             sample[v] = val
    ...
    ...         response.add_sample(sample, Q=Q)
    ...         return response

    This will now behave as expected

    >>> Q = {(0, 0): 1, (1, 1): 0}
    >>> response = MyLinearSampler().sample_qubo(Q)
    >>> list(response.samples())
    [{0: 0, 1: 1}]

    Also, by implementing one of the methods, we now can use the others.

    >>> h = {0: -1, 1: 2}
    >>> J = {}
    >>> response = MyLinearSampler().sample_ising(h, J)
    >>> list(response.samples())
    [{0: 1, 1: -1}]

    Similarily for the structured methods.

    >>> h = {0: -1, 1: 2}
    >>> J = {}
    >>> response = MyLinearSampler().sample_structured_ising(h, J)
    >>> list(response.samples())
    [{0: 1, 1: -1}]
    >>> Q = {(0, 0): 1, (1, 1): 0}
    >>> response = MyLinearSampler().sample_structured_qubo(Q)
    >>> list(response.samples())
    [{0: 0, 1: 1}]

    However, if we assign a structure to our sampler, the structured
    methods will no longer work.

    >>> sampler = MyLinearSampler()
    >>> sampler.structure = 'linear'
    >>> try:
    ...     sampler.sample_structured_qubo({})
    ... except NotImplementedError:
    ...     print('not implemented')
    'not implemented'

"""


from dimod.decorators import ising, qubo
from dimod.utilities import qubo_to_ising, ising_to_qubo

__all__ = ['TemplateSampler']


class TemplateSampler(object):
    """Serves as a template for samplers. Not intended to be used directly.

    The methods as provided are self-referential, trying to invoke them
    directly will lead to an infinite recursion. This is done so that users
    need only implement the methods that make sense.

    See module documentation for examples.
    """
    def __init__(self):
        self.structure = None

    @qubo(1)
    def sample_qubo(self, Q, **kwargs):
        """Converts the given QUBO into an Ising problem, then invokes the
        sample_ising method.

        See sample_ising documentation for more information.

        Args:
            Q (dict): A dictionary defining the QUBO. Should be of the form
                {(u, v): bias} where u, v are variables and bias is numeric.
            **kwargs: Any keyword arguments are passed directly to
                sample_ising.

        Returns:
            :obj:`BinaryResponse`:
                A `BinaryResponse`, converted from the `SpinResponse` return
                from sample_ising.

        Note:
            This method is inherited from the :obj:`TemplateSampler` base class.

        """
        h, J, offset = qubo_to_ising(Q)
        spin_response = self.sample_ising(h, J, **kwargs)
        return spin_response.as_binary(offset)

    @ising(1, 2)
    def sample_ising(self, h, J, **kwargs):
        """Converts the given Ising probkem into a QUBO, then invokes the
        sample_qubo method.

        See sample_qubo documentation for more information.

        Args:
            h (dict/list): The linear terms in the Ising problem. If a
                dict, should be of the form {v: bias, ...} where v is
                a variable in the Ising problem, and bias is the linear
                bias associated with v. If a list, should be of the form
                [bias, ...] where the indices of the biases are the
                variables in the Ising problem.
            J (dict): A dictionary of the quadratic terms in the Ising
                problem. Should be of the form {(u, v): bias} where u,
                v are variables in the Ising problem and bias is the
                quadratic bias associated with u, v.
            **kwargs: Any keyword arguments are passed directly to
                sample_qubo.

        Returns:
            :obj:`SpinResponse`:
                A `SpinResponse`, converted from the `BinaryResponse`
                return from sample_ising.

        Note:
            This method is inherited from the :obj:`TemplateSampler` base class.

        """
        Q, offset = ising_to_qubo(h, J)
        binary_response = self.sample_qubo(Q, **kwargs)
        return binary_response.as_spin(offset)

    @qubo(1)
    def sample_structured_qubo(self, Q, **kwargs):
        """Invokes the sample_qubo method.

        See sample_qubo documentation for more information.

        Args:
            Q (dict): A dictionary defining the QUBO. Should be of the form
                {(u, v): bias} where u, v are variables and bias is numeric.
            **kwargs: Any keyword arguments are passed directly to
                sample_ising.

        Returns:
            :obj:`BinaryResponse`:
                A `BinaryResponse`, converted from the `SpinResponse` return
                from sample_ising.

        Note:
            This method is inherited from the :obj:`TemplateSampler` base class.

        Raises:
            NotImplementedError: If the `structure` property is not None.

        """
        if self.structure is not None:
            raise NotImplementedError()
        return self.sample_qubo(Q, **kwargs)

    @ising(1, 2)
    def sample_structured_ising(self, h, J, **kwargs):
        """Invokes the sample_ising method.

        See sample_qubo documentation for more information.

        Args:
            h (dict/list): The linear terms in the Ising problem. If a
                dict, should be of the form {v: bias, ...} where v is
                a variable in the Ising problem, and bias is the linear
                bias associated with v. If a list, should be of the form
                [bias, ...] where the indices of the biases are the
                variables in the Ising problem.
            J (dict): A dictionary of the quadratic terms in the Ising
                problem. Should be of the form {(u, v): bias} where u,
                v are variables in the Ising problem and bias is the
                quadratic bias associated with u, v.
            **kwargs: Any keyword arguments are passed directly to
                sample_qubo.

        Returns:
            :obj:`SpinResponse`:
                A `SpinResponse`, converted from the `BinaryResponse`
                return from sample_ising.

        Note:
            This method is inherited from the :obj:`TemplateSampler` base class.

        Raises:
            NotImplementedError: If the `structure` property is not None.

        """
        if self.structure is not None:
            raise NotImplementedError()
        return self.sample_ising(h, J, **kwargs)

    @property
    def structure(self):
        return self._structure

    @structure.setter
    def structure(self, struct):
        self._structure = struct
