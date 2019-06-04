Development environment
=======================

A development environment can be installed using `pipenv <https://pipenv.readthedocs.io/>`_.
Running ``pipenv install --dev`` creates a virtual environment with python 3.6,
installs all the dependencies and the package spellvardetection in editable
mode so that it can be used for development.

The repository contains a Makefile that runs common steps like testing and
building the documentation using the pipenv environment.

Tests
=====

Tests are run using `nose2 <https://nose2.readthedocs.io>`_, with `tox
<https://tox.readthedocs.io>`_ handling tests for multiple python versions (3.6
and 3.7). tox installs dependencies from requirements.txt into its virtual
environments.

When using pipenv ``make test`` first updates requirements.txt based on
Pipfile.lock before running tox, adding the --recreate flag if the requirements
have changed - so the tests are run in virtual environments with the same setup
as the main pipenv environment.

To get coverage statistics, run ``make coverage``.

Documentation
=============

The documentation is built using `sphinx <https://www.sphinx-doc.org/>`_.

When using pipenv sphinx and other packages that are needed are installed into
the development environment. Running ``make docs`` creates the html
documentation using pipenv.
