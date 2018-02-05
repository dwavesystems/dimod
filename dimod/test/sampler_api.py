class SamplerAPITest:
    """Provides a series of generic API tests that all samplers should pass.
    """
    def test_instantiation(self):
        sampler = self.sampler_factory()
