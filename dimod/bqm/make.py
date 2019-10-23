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
import contextlib
import os


@contextlib.contextmanager
def change_cwd(loc):
    """Temporarily change working directory to the given location."""
    cwd = os.getcwd()

    os.chdir(loc)
    try:
        yield
    finally:
        # return to original location
        os.chdir(cwd)


header = "\"This file is generated. See sheapeablebqm.pyx.src and make.py\""


def make_bqms(loc):
    """`loc` is the directory containing shapeablebqm.pyx.src"""
    tname = 'shapeablebqm.pyx.src'

    with change_cwd(loc):
        template_mtime = os.path.getmtime(tname)

        with open('shapeablebqm.pyx.src', 'r') as fp:
            template = fp.read()

        for type_name in ['Vector', 'Map']:
            fname = 'adj{}bqm.pyx'.format(type_name.lower())

            # Cython won't recompile files that have not been modified, so we do
            # the same, checking if it's last modification is older than the
            # modification time of the template file. This check can fail if the
            # generated file has been modified by-hand, but in order to keep the
            # logic simple we'll ignore that case.
            if os.path.isfile(fname) and os.path.getmtime(fname) > template_mtime:
                continue

            print(("Generating dimod/bqm/{} because dimod/bqm/{} changed."
                   ).format(fname, tname))

            contents = template
            contents = contents.replace('@header@', header)
            contents = contents.replace('@name@', type_name)

            with open(fname, 'w') as fp:
                fp.write(contents)


if __name__ == '__main__':
    make_bqms(os.path.dirname(os.path.abspath(__file__)))
