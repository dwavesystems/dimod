# Copyright 2019 D-Wave Systems Inc.
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
# =============================================================================
import numpy as np

itype = np.dtype(np.uint32)  # corresponds to VarIndex
dtype = np.dtype(np.float64)  # corresponds to Bias

# numpy does not have a dtype corresponding to size_t, normally uint64
ntype = np.dtype(('u' if ((<size_t>-1) > 0) else 'i') + str(sizeof(size_t)))
