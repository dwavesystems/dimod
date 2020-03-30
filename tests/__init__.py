# Copyright 2020 D-Wave Systems Inc.
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

# developer note: we could mess around with discovery, but this is much simpler
try:
    from tests.test_cppbqm import *
except ImportError:
    print("tests/test_cppbqm.pyx is not built or discoverable")

try:
    from tests.test_cybqm import *
except ImportError:
    print("tests/test_cybqm.pyx is not built or discoverable")
