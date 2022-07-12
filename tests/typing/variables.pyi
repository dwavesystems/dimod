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

# developer note: this is far from complete, but tests that the Cython
# code is getting annotated correctly

from dimod.variables import Variables


Variables('abc')
Variables(object())  # E: Argument 1 to "Variables" has incompatible type

a: str = Variables().is_range  # E: Incompatible types
b: bool = Variables().is_range

v: Variables = Variables().copy()
