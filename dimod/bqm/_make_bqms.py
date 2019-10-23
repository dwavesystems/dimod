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
"""Generate the .pyx files for AdjVectorBQM and AdjMapBQM."""
import os


def _make():
    header = "\"This file is generated. See sheapeablebqm.pyx.src and _make_bqms.py\""

    # run from the same location as the file
    my_loc = os.path.dirname(os.path.abspath(__file__))
    os.chdir(my_loc)

    tname = 'shapeablebqm.pyx.src'
    template_mtime = os.path.getmtime(tname)

    with open('shapeablebqm.pyx.src', 'r') as fp:
        template = fp.read()

    matrix = [dict(name='Vector',
                   ),
              dict(name='Map',
                   ),
              ]

    for typ in matrix:
        fname = 'adj{}bqm.pyx'.format(typ['name'].lower())

        # cython won't recompile files that have not been modified, so we do
        # the same, checking if it's last modification is older than the
        # modification time of the template file. This check can fail if the
        # generated file has been modified by-hand, but in order to keep the
        # logic simple we'll ignore that case.
        if os.path.getmtime(fname) > template_mtime:
            continue

        contents = template
        contents = contents.replace('@header@', header)
        contents = contents.replace('@name@', typ['name'])

        with open(fname, 'w') as fp:
            fp.write(contents)


if __name__ == '__main__':
    _make()
