class InvalidComposition(Exception):
    """Used for compositions of samplers that are invalid"""


class MappingError(Exception):
    """mapping causes conflicting values in samples"""


class InvalidSampler(Exception):
    """Raised when trying to use Sampler as a sampler"""
