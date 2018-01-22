"""
Conversion functions
--------------------

These functions convert the :class:`.BinaryQuadraticModel` to other datatypes.

"""

from dimod.compatibility23 import iteritems
from dimod.binary_quadratic_model.model import BinaryQuadraticModel
from dimod.vartypes import Vartype

__all__ = ['to_networkx_graph',
           'to_ising',
           'to_qubo',
           'to_numpy_matrix',
           'to_pandas_dataframe',
           'from_ising',
           'from_qubo',
           'from_numpy_matrix',
           'from_pandas_dataframe']


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


def from_ising(h, J, offset=0.0):
    """Build a binary quadratic model from an Ising problem.


    Args:
        h (dict[variable, bias]/list[bias]):
            The linear biases of the Ising problem. If a list, the indices of the list are treated
            as the variable labels.

        J (dict[(variable, variable), bias]):
            The quadratic biases of the Ising problem.

        offset (optional, default=0.0):
            The constant offset applied to the model.

    Returns:
        :class:`.BinaryQuadraticModel`

    """
    if isinstance(h, list):
        h = dict(enumerate(h))

    return BinaryQuadraticModel(h, J, offset, Vartype.SPIN)


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


def from_qubo(Q, offset=0.0):
    """Build a binary quadratic model from a qubo.

    Args:
        Q (dict):
            The qubo coefficients.

        offset (optional, default=0.0):
            The constant offset applied to the model.

    Returns:
        :class:`.BinaryQuadraticModel`

    """
    linear = {}
    quadratic = {}
    for (u, v), bias in iteritems(Q):
        if u == v:
            linear[u] = bias
        else:
            quadratic[(u, v)] = bias

    return BinaryQuadraticModel(linear, quadratic, offset, Vartype.BINARY)


def to_numpy_matrix(bqm, variable_order=None):
    """Return the binary quadratic model as a matrix.

    Args:
        bqm (:class:`.BinaryQuadraticModel`):
            A binary quadratic model. Should either be index-labeled from 0 to N-1 or variable_order
            should be provided.

        variable_order (list, optional):
            If variable_order is provided, the rows/columns of the numpy array are indexed by
            the variables in variable_order. If any variables are included in variable_order that
            are not in `bqm`, they will be included in the matrix.

    Returns:
        :class:`numpy.matrix`: The binary quadratic model as a matrix. The matrix has binary
        vartype.

    Notes:
        The matrix representation of a binary quadratic model only makes sense for binary models.
        For a binary sample x, the energy of the model is given by:

        .. math::

            E(x) = x^T Q x


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

    return np.asmatrix(mat)


def from_numpy_matrix(mat, variable_order=None, offset=0.0, interactions=[]):
    """Build a binary quadratic model from a numpy matrix.

    Args:
        mat (:class:`numpy.matrix`):
            A square numpy matrix. The coefficients of a qubo.

        variable_order (list, optional):
            If variable_order is provided, provides the labels for the variables in the binary
            quadratic program, otherwise the row/column indices will be used. If variable_order
            is longer than the matrix, the extra values are ignored.

        offset (optional, default=0.0):
            The constant offset for the binary quadratic program.

        interactions (iterable, optional, default=[]):
            Any additional 0.0-bias interactions to be added to the binary quadratic model.

    Returns:
        :class:`.BinaryQuadraticModel`

    """
    import numpy as np

    if mat.ndim != 2:
        raise ValueError("expected input mat to be a square matrix")  # pragma: no cover

    num_row, num_col = mat.shape
    if num_col != num_row:
        raise ValueError("expected input mat to be a square matrix")  # pragma: no cover

    if variable_order is None:
        variable_order = list(range(num_row))

    bqm = BinaryQuadraticModel({}, {}, offset, Vartype.BINARY)

    for (row, col), bias in np.ndenumerate(mat):
        if row == col:
            bqm.add_variable(variable_order[row], bias)
        elif bias:
            bqm.add_interaction(variable_order[row], variable_order[col], bias)

    for u, v in interactions:
        bqm.add_interaction(u, v, 0.0)

    return bqm


def to_pandas_dataframe(bqm):
    """Return the binary quadratic model as a pandas DataFrame.

    Args:
        bqm (:class:`.BinaryQuadraticModel`):
            A binary quadratic model. Should either be index-labeled from 0 to N-1 or variable_order
            should be provided.

    Returns:
        :class:`pandas.DataFrame`: The binary quadratic model as a DataFrame. The DataFrame has
        binary vartype. The rows and columns are labeled by the variables in the binary quadratic
        model.

    Notes:
        The DataFrame representation of a binary quadratic model only makes sense for binary models.
        For a binary sample x, the energy of the model is given by:

        .. math::

            E(x) = x^T Q x


        The offset is dropped when converting to a pandas DataFrame.

    """
    import pandas as pd

    try:
        variable_order = sorted(bqm.linear)
    except TypeError:
        variable_order = list(bqm.linear)

    return pd.DataFrame(to_numpy_matrix(bqm, variable_order=variable_order),
                        index=variable_order,
                        columns=variable_order)  # let it choose its own datatype


def from_pandas_dataframe(bqm_df, offset=0.0, interactions=[]):
    """Build a binary quadratic model from a pandas dataframe.

    Args:
        bqm_df (:class:`pandas.DataFrame`):
            A pandas dataframe. The row and column indices should be that variables of the binary
            quadratic program. The values should be the coefficients of a qubo.

        offset (optional, default=0.0):
            The constant offset for the binary quadratic program.

        interactions (iterable, optional, default=[]):
            Any additional 0.0-bias interactions to be added to the binary quadratic model.

    Returns:
        :class:`.BinaryQuadraticModel`

    """
    bqm = BinaryQuadraticModel({}, {}, offset, Vartype.BINARY)

    for u, row in bqm_df.iterrows():
        for v, bias in row.iteritems():
            if u == v:
                bqm.add_variable(u, bias)
            elif bias:
                bqm.add_interaction(u, v, bias)

    for u, v in interactions:
        bqm.add_interaction(u, v, 0.0)

    return bqm
