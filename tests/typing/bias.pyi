# Copyright 2022 D-Wave Systems Inc.
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

import numpy as np

from dimod.typing import Bias


# built-in numeric types
a: Bias = 1
b: Bias = 1.
c: Bias = complex(1, 1)  # E: Incompatible types in assignment

# built-in non-numeric
r: Bias = 'hello'  # E: Incompatible types in assignment
s: Bias = [0, 1]  # E: Incompatible types in assignment

# NumPy numeric types
j: Bias = np.int64(1)
k: Bias = np.int8(1)
l: Bias = np.float32(1)
m: Bias = np.float64(1)
n: Bias = np.complex64(5)  # E: Incompatible types in assignment
