class InvalidComposition(Exception):
    """Used for compositions of samplers that are invalid"""


class MappingError(Exception):
    """mapping causes conflicting values in samples"""


class InvalidSampler(Exception):
    """Raised when trying to use Sampler as a sampler"""


class BinaryQuadraticModelValueError(ValueError):
    """Raised when a sampler cannot handle a specific bqm"""


class BinaryQuadraticModelSizeError(BinaryQuadraticModelValueError):
    """Raised when the bqm has too many variables"""


class BinaryQuadraticModelStructureError(BinaryQuadraticModelValueError):
    """Raised when the BQM does not fit the sampler"""
