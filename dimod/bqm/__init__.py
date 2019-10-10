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
import sys

from dimod.bqm.adjdictbqm import AdjDictBQM

if sys.version_info.major == 3 and sys.version_info.minor >= 5:
    from dimod.bqm.adjarraybqm import AdjArrayBQM
    from dimod.bqm.adjmapbqm import AdjMapBQM
    from dimod.bqm.adjvectorbqm import AdjVectorBQM

del sys  # so not in namespace
