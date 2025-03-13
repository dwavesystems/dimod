# Configuration file for the Sphinx documentation builder.
#
# This file only contains a selection of the most common options. For a full
# list see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html
#
import os
import subprocess
import sys

config_directory = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.abspath('.'))
sys.path.insert(0, os.path.dirname(config_directory))

# -- General configuration ------------------------------------------------

extensions = [
    'sphinx.ext.autosummary',
    'sphinx.ext.autodoc',
    'sphinx.ext.coverage',
    'sphinx.ext.doctest',
    'sphinx.ext.intersphinx',
    'sphinx.ext.mathjax',
    'sphinx.ext.napoleon',
    'sphinx.ext.todo',
    'sphinx.ext.viewcode',
    'sphinx.ext.ifconfig',
    'breathe',
    'reno.sphinxext',
]

autosummary_generate = True


source_suffix = ['.rst']

master_doc = 'index'

# TODO: delete unneeded parts of this file

# General information about the project.
project = u'dimod'
copyright = u'2017, D-Wave Systems Inc'
author = u'D-Wave Systems Inc'


import dimod
version = dimod.__version__
release = dimod.__version__

language = "en"

add_module_names = False

exclude_patterns = ['build', 'Thumbs.db', '.DS_Store', 'sdk_index.rst']

linkcheck_retries = 2
linkcheck_anchors = False
linkcheck_ignore = [r'https://cloud.dwavesys.com/leap',  # redirects, many checks
                    r'https://docs.ocean.dwavesys.com/projects/dimod',
                    r'https://epubs.siam.org',           # ignores robots since Feb 2023
                    r'.clang-format',
                    r'setup.cfg',
                    ]  # dimod RTD currently not building on boost

pygments_style = 'sphinx'

todo_include_todos = True

modindex_common_prefix = ['dimod.']

doctest_global_setup = """
from __future__ import print_function, division

import dimod

"""

# -- Breath ---------------------------------------------------------------

breathe_default_project = "dimod"
breathe_projects = dict(
  dimod=os.path.join(config_directory, 'build-cpp', 'xml'),
  )

# see https://breathe.readthedocs.io/en/latest/readthedocs.html
if os.environ.get('READTHEDOCS', False):
    subprocess.call('make cpp', shell=True, cwd=config_directory)

# -- Options for HTML output ----------------------------------------------
html_theme = "pydata_sphinx_theme"
html_theme_options = {
    "collapse_navigation": True,
    "show_prev_next": False,
}
html_sidebars = {"**": ["search-field", "sidebar-nav-bs"]}  # remove ads

intersphinx_mapping = {'python': ('https://docs.python.org/3', None),
                       'numpy': ('https://numpy.org/doc/stable/', None),
                       'networkx': ('https://networkx.org/documentation/stable/', None),
                       'dwave': ('https://docs.dwavequantum.com/en/latest/', None),
                       }
