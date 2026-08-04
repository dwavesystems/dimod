#!/usr/bin/env python3

# Copyright 2024 D-Wave Inc.
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

if __name__ == "__main__":
    import os.path
    import re

    pck_root = os.path.dirname(os.path.dirname(__file__))
    with open(os.path.join(pck_root, '__init__.py')) as f:
        m = re.search(
            r"__version__ = \"([0-9]+(\.[0-9]+)*((\.dev|rc)([0-9]+)?)?)\"",
            f.read()
        )
    print(m.group(1))
