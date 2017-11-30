class SamplerKeywordArg(object):
    """Allows for passing information about keyword parameters accepted
    by dimod samplers.

    Args:
        name (str): The keyword argument's name.
        type_annotation (str): The keyword type as a string.
        classinfo (type): The expected type of the parameter.

    Attributes:
        name (str): The keyword argument's name.
        type_annotation (str): The keyword type as a string.
        classinfo (type): The expected type of the parameter.

    Examples:
        >>> num_samples_kwarg = SamplerKeywordArg('num_samples', 'int', int)
        >>> embedding_kwarg = SamplerKeywordArg('embedding', 'dict[hashable, iterable]', dict)

    """
    # it would be cool if isinstance(6, SamplerKeywordArg('num_samples', 'int', int)) => True
    # if you can get it to work then please do!

    def __init__(self, name, type_annotation='', classinfo=object):
        if not isinstance(name, str):
            raise TypeError("expected input 'name' to be a str.")
        self.name = name

        if not isinstance(type_annotation, str):
            raise TypeError("expected input 'type_annotation' to be a str.")
        self.type_annotation = type_annotation

        if not isinstance(classinfo, type):
            raise TypeError("expected input 'classinfo' to be a type.")
        self.classinfo = classinfo

    def __str__(self):
        return "SamplerKeywordArg('{}', '{}', {})".format(self.name, self.type_annotation, self.classinfo)
