class InvalidComposition(Exception):
    """Raised for compositions of samplers that are invalid"""


class MappingError(Exception):
    """Raised when mapping causes conflicting values in samples"""


class InvalidSampler(Exception):
    """Raised when trying to use the specified sampler as a sampler"""


class BinaryQuadraticModelValueError(ValueError):
    """Raised when a sampler cannot handle a specified binary quadratic model"""


class BinaryQuadraticModelSizeError(BinaryQuadraticModelValueError):
    """Raised when the binary quadratic model has too many variables"""


class BinaryQuadraticModelStructureError(BinaryQuadraticModelValueError):
    """Raised when the binary quadratic model does not fit the sampler"""
