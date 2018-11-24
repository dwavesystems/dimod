"""
Used to generate the .pyx files for the different vectors.
"""
import os

# run from the same location as the file
my_loc = os.path.dirname(os.path.abspath(__file__))
os.chdir(my_loc)

# get the template
with open('vector.pyx.template', 'r') as fp:
    template = fp.read()

# for types see https://docs.scipy.org/doc/numpy-1.15.0/user/basics.types.html#data-types
# for format see https://docs.python.org/3.7/library/array.html
datatypes = [('npy_float32', 'f'),
             ('npy_float64', 'd'),
             ('npy_int8', 'b'),
             ('npy_int16', 'h'),
             ('npy_int32', 'i'),  # todo: this needs to be tested on 32-bit systems
             ('npy_int64', 'q'),  # todo: this might be too large for some systems, also probably unnessary for us
             ]

for dtype, form in datatypes:

    fname = 'vector_{dtype}.pyx'.format(dtype=dtype)
    contents = template.format(dtype=dtype, format=form)

    with open(fname, 'w') as fp:
        fp.write(contents)
