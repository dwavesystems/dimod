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


def to_networkx_graph(model, node_attribute_name='bias', edge_attribute_name='bias'):
    """Return the BinaryQuadraticModel as a NetworkX graph.

    Args:
        node_attribute_name (hashable): The attribute name for the
            linear biases.
        edge_attribute_name (hashable): The attribute name for the
            quadratic biases.

    Returns:
        :class:`networkx.Graph`: A NetworkX with the biases stored as
        node/edge attributes.

    Examples:
        >>> import networkx as nx
        >>> model = pm.BinaryQuadraticModel({0: 1, 1: -1, 2: .5},
        ...                                 {(0, 1): .5, (1, 2): 1.5},
        ...                                 1.4,
        ...                                 pm.SPIN)
        >>> BQM = model.to_networkx_graph()
        >>> BQM[0][1]['bias']
        0.5


    """
    import networkx as nx

    BQM = nx.Graph()

    # add the linear biases
    BQM.add_nodes_from(((v, {node_attribute_name: bias, 'vartype': model.vartype})
                        for v, bias in iteritems(model.linear)))

    # add the quadratic biases
    BQM.add_edges_from(((u, v, {edge_attribute_name: bias}) for (u, v), bias in iteritems(model.quadratic)))

    # set the offset
    BQM.offset = model.offset

    return BQM


def to_ising(model):
    """Converts the model into the (h, J, offset) Ising format.

    If the model type is not spin, it is converted.

    Returns:
        tuple: A 3-tuple:

            dict: The linear biases.

            dict: The quadratic biases.

            number: The offset.

    """
    if model.vartype == model.SPIN:
        # can just return the model as-is.
        return model.linear, model.quadratic, model.offset

    if model.vartype != model.BINARY:
        raise RuntimeError('converting from unknown vartype')

    return model.binary_to_spin(model.linear, model.quadratic, model.offset)


def to_qubo(model):
    """Converts the model into the (Q, offset) QUBO format.

    If the model type is not binary, it is converted.

    Returns:
        tuple: A 2-tuple:

            dict: The qubo biases.

            number: The offset.

    """
    if model.vartype == model.BINARY:
        # need to dump the linear biases into quadratic
        qubo = {}
        for v, bias in iteritems(model.linear):
            qubo[(v, v)] = bias
        for edge, bias in iteritems(model.quadratic):
            qubo[edge] = bias
        return qubo, model.offset

    if model.vartype != model.SPIN:
        raise RuntimeError('converting from unknown vartype')

    linear, quadratic, offset = model.spin_to_binary(model.linear, model.quadratic, model.offset)

    quadratic.update({(v, v): bias for v, bias in iteritems(linear)})

    return quadratic, offset
