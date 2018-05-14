import re

from dimod.binary_quadratic_model import BinaryQuadraticModel

_LINE_REGEX = r'^\s*(\d+)\s+(\d+)\s+([+-]?([0-9]*[.])?[0-9]+)\s*$'
"""
Each line should look like

0 1 2.000000

where 0, 1 are the variable lables and 2.0 is the bias.
"""


def dumps(bqm):
    """Dump a binary quadratic model to a string in COOrdinate format."""
    return '\n'.join(_iter_triplets(bqm))


def dump(bqm, fp):
    """Dump a binary quadratic model to a string in COOrdinate format."""
    for triplet in _iter_triplets(bqm):
        fp.write('%s\n' % triplet)


def loads(s, bqm):
    """Load a COOrdinate formatted binary quadratic model from a string."""
    pattern = re.compile(_LINE_REGEX)

    for line in s.split('\n'):
        match = pattern.search(line)

        if match is not None:
            u, v, bias = int(match.group(1)), int(match.group(2)), float(match.group(3))
            if u == v:
                bqm.add_variable(u, bias)
            else:
                bqm.add_interaction(u, v, bias)

    return bqm


def load(fp, bqm):
    """Load a COOrdinate formatted binary quadratic model from a file."""
    pattern = re.compile(_LINE_REGEX)

    for line in fp:
        match = pattern.search(line)

        if match is not None:
            u, v, bias = int(match.group(1)), int(match.group(2)), float(match.group(3))
            if u == v:
                bqm.add_variable(u, bias)
            else:
                bqm.add_interaction(u, v, bias)

    return bqm


def _iter_triplets(bqm):
    if not isinstance(bqm, BinaryQuadraticModel):
        raise TypeError("expected input to be a BinaryQuadraticModel")
    if not all(isinstance(v, int) and v >= 0 for v in bqm.linear):
        raise ValueError("only positive index-labeled binary quadratic models can be dumped to COOrdinate format")

    # developer note: we could (for some threshold sparseness) sort the neighborhoods,
    # but this is simple and probably sufficient

    variables = sorted(bqm)  # integer labeled so we can sort in py3

    for idx, u in enumerate(variables):
        for v in variables[idx:]:
            if u == v and bqm.linear[u]:
                yield '%d %d %f' % (u, u, bqm.linear[u])
            elif u in bqm.adj[v]:
                yield '%d %d %f' % (u, v, bqm.adj[u][v])
