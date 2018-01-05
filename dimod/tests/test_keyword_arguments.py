import unittest

from dimod import SamplerKeywordArg


class TestSamplerKeywordArg(unittest.TestCase):
    def test_docstring(self):
        num_samples_kwarg = SamplerKeywordArg('num_samples', 'int', int)
        embedding_kwarg = SamplerKeywordArg('embedding', 'dict[hashable, iterable]', dict)

    def test_isinstancec(self):
        num_samples_kwarg = SamplerKeywordArg('num_samples', 'int', int)

        self.assertTrue(isinstance(6, num_samples_kwarg.classinfo))

    def test_string(self):
        num_samples_kwarg = SamplerKeywordArg('num_samples', 'int', int)
        self.assertIsInstance(num_samples_kwarg.__str__(), str)

    def test_exceptions(self):
        with self.assertRaises(TypeError):
            num_samples_kwarg = SamplerKeywordArg('num_samples', 'int', 'int')

        with self.assertRaises(TypeError):
            num_samples_kwarg = SamplerKeywordArg('num_samples', int, int)

        with self.assertRaises(TypeError):
            num_samples_kwarg = SamplerKeywordArg(100, 'int', int)

    def test_unicode(self):
        # should allow
        SamplerKeywordArg(u'hello', 'str', str)
