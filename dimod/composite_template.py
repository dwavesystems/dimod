"""
Samplers can be composed. This pattern allows pre- and post-processing
to be applied to binary quadratic programs without needing to change
the underlying sampler implementation.

Examples
--------



"""


class TemplateComposite(object):
    """Serves as a template for composits. Not intended to be used directly.

    Attributes:
        children (list): A list of child samplers or child composed samplers.

    """
    def __init__(self):
        self.children = []
