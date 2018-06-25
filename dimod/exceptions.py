# Copyright 2018 D-Wave Systems Inc.
#
#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at
#
#        http://www.apache.org/licenses/LICENSE-2.0
#
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.
#
# ================================================================================================


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
