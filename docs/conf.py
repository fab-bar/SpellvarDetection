# -*- coding: utf-8 -*-
#
# Configuration file for the Sphinx documentation builder.
#
# This file does only contain a selection of the most common options. For a
# full list see the documentation:
# http://www.sphinx-doc.org/en/master/config

# -- Project information -----------------------------------------------------

project = 'SpellvarDetection'
copyright = '2018-2019, Fabian Barteld'
author = 'Fabian Barteld'

# get the version
# this needs the package to be installed
from pkg_resources import get_distribution
release = get_distribution('spellvardetection').version
version = '.'.join(release.split('.')[:2])


# -- General configuration ---------------------------------------------------

extensions = [
    'm2r',
]

# The suffix(es) of source filenames.
source_suffix = '.rst'

# The master toctree document.
master_doc = 'index'

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']


# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.
html_theme = 'sphinx_rtd_theme'

# -- Options for LaTeX output ------------------------------------------------

latex_elements = {
    'printindex': ''
}

# Grouping the document tree into LaTeX files. List of tuples
# (source start file, target name, title,
#  author, documentclass [howto, manual, or own class]).
latex_documents = [
    (master_doc, 'SpellvarDetection.tex', 'SpellvarDetection Documentation',
     'Fabian Barteld', 'manual'),
]
