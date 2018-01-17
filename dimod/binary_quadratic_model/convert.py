from dimod import _PY2
from dimod.vartypes import Vartype

if _PY2:
    def iteritems(d):
        return d.iteritems()

    def itervalues(d):
        return d.itervalues()

else:
    def iteritems(d):
        return d.items()

    def itervalues(d):
        return d.values()


def to_networkx_graph(bqm, node_attribute_name='bias', edge_attribute_name='bias'):
    """Return the BinaryQuadraticModel as a NetworkX graph.

    Args:
        node_attribute_name (hashable):
            The attribute name for the linear biases.
        edge_attribute_name (hashable):
            The attribute name for the quadratic biases.

    Returns:
        :class:`networkx.Graph`: A NetworkX with the biases stored as
        node/edge attributes.

    Examples:
        >>> import networkx as nx
        >>> bqm = dimod.BinaryQuadraticModel({0: 1, 1: -1, 2: .5},
        ...                                  {(0, 1): .5, (1, 2): 1.5},
        ...                                  1.4,
        ...                                  dimod.SPIN)
        >>> BQM = dimod.to_networkx_graph(bqm)
        >>> BQM[0][1]['bias']
        0.5
        >>> BQM.node[0]['bias']
        1

        Also, if the preferred notation is 'weights'

        >>> import networkx as nx
        >>> bqm = dimod.BinaryQuadraticModel({0: 1, 1: -1, 2: .5},
        ...                                  {(0, 1): .5, (1, 2): 1.5},
        ...                                  1.4,
        ...                                  dimod.SPIN)
        >>> BQM = dimod.to_networkx_graph(bqm, edge_attribute_name='weight')
        >>> BQM[0][1]['weight']
        0.5


    """
    import networkx as nx

    BQM = nx.Graph()

    vartype = bqm.vartype

    # add the linear biases
    BQM.add_nodes_from(((v, {node_attribute_name: bias, 'vartype': vartype})
                        for v, bias in iteritems(bqm.linear)))

    # add the quadratic biases
    BQM.add_edges_from(((u, v, {edge_attribute_name: bias}) for (u, v), bias in iteritems(bqm.quadratic)))

    # set the offset and vartype properties for the graph
    BQM.offset = bqm.offset
    BQM.vartype = vartype

    return BQM


def to_ising(bqm):
    """Converts the binary quadratic model into the (h, J, offset) Ising format.

    If the binary quadratic model's vartype is not spin, it is converted.

    Args:
        bqm (:class:`.BinaryQuadraticModel`):
            A binary quadratic model.

    Returns:
        tuple: A 3-tuple:

            dict: The linear biases.

            dict: The quadratic biases.

            number: The offset.

    """
    return bqm.spin.linear, bqm.spin.quadratic, bqm.spin.offset


def to_qubo(bqm):
    """Converts the binary quadratic model into the (Q, offset) QUBO format.

    If the binary quadratic model's vartype is not binary, it is converted.

    Args:
        bqm (:class:`.BinaryQuadraticModel`):
            A binary quadratic model.

    Returns:
        tuple: A 2-tuple:

            dict: The qubo biases. A dict where the keys are pairs of variables and the values
            are the associated linear or quadratic bias.

            number: The offset.

    """
    qubo = {}

    for v, bias in iteritems(bqm.binary.linear):
        qubo[(v, v)] = bias

    for edge, bias in iteritems(bqm.binary.quadratic):
        qubo[edge] = bias

    return qubo, bqm.binary.offset


def to_numpy_array(bqm, variable_order=None):
    """Return the binary quadratic model as a matrix.

    Args:
        bqm (:class:`.BinaryQuadraticModel`):
            A binary quadratic model. Should either be index-labeled from 0 to N-1 or variable_order
            should be provided.

        variable_order (list, optional):
            If variable_order is provided, the rows/columns of the numpy array are indexed by
            the variables in variable_order.

    Returns:
        :class:`numpy.ndarray`: The binary quadratic model as a matrix. The matrix has binary
        vartype.

    Notes:
        The matrix representation of a binary quadratic model only makes sense for binary models.
        For a binary sample x, the energy of the model is given by:

        .. math::

            x^T Q x


        The offset is dropped when converting to a numpy matrix.


    """
    import numpy as np

    if variable_order is None:
        # just use the existing variable labels, assuming that they are [0, N)
        num_variables = len(bqm)
        mat = np.zeros((num_variables, num_variables), dtype=float)

        try:
            for v, bias in iteritems(bqm.binary.linear):
                mat[v, v] = bias
        except IndexError:
            raise ValueError(("if 'variable_order' is not provided, binary quadratic model must be "
                              "index labeled [0, ..., N-1]"))

        for (u, v), bias in iteritems(bqm.binary.quadratic):
            if u < v:
                mat[u, v] = bias
            else:
                mat[v, u] = bias

    else:
        num_variables = len(variable_order)
        idx = {v: i for i, v in enumerate(variable_order)}

        mat = np.zeros((num_variables, num_variables), dtype=float)

        try:
            for v, bias in iteritems(bqm.binary.linear):
                mat[idx[v], idx[v]] = bias
        except KeyError as e:
            raise ValueError(("variable {} is missing from variable_order".format(e)))

        for (u, v), bias in iteritems(bqm.binary.quadratic):
            iu, iv = idx[u], idx[v]
            if iu < iv:
                mat[iu, iv] = bias
            else:
                mat[iv, iu] = bias

    return mat
