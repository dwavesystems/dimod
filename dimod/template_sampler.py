"""
The sampler template provides an API that different samplers can use.
The crux of the API are two methods and a property.

Methods:

- sample_ising
- sample_qubo

Property:

- structure

Some samplers will only function on binary quadratic models with specific
structures. These samplers should have information in the `structure`
property. See Structured Samplers section below.


Examples
--------
Define a sampler that operates on QUBO problems:

>>> class MyLinearSampler(dimod.TemplateSampler):
...     def __init__(self):
...         dimod.TemplateSampler.__init__(self)
...         self.structure = 'linear'
...
...     @dimod.decorators.qubo(1)
...     def sample_qubo(self, Q):
...         response = dimod.BinaryResponse()
...
...         sample = {}
...         for (u, v) in Q:
...             if u != v:
...                 raise ValueError()
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


Structured Samplers
-------------------

Some samplers can only operate on a particular problem structure.
Most commonly this happens when there is a particular problem graph.
In this case, the `structure` property will not be None.

"""
# we could do a solver that requires a complete graph as an example above...


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
        """Converts the given Ising problem into a QUBO, then invokes the
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

    @property
    def structure(self):
        """Structure for the sampler. None for unstructured samplers."""
        return self._structure

    @structure.setter
    def structure(self, struct):
        self._structure = struct
