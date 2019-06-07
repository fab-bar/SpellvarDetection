Getting started
===============

Please note that this software package is still under development and the user
interface will likely change.

Installation
------------

The latest version of SpellvarDectection can be downloaded from `GitHub
<https://github.com/fab-bar/SpellvarDetection/releases>`_. To use it `Python 3.6
or 3.7 <https://www.python.org/downloads/>`_ needs to be installed on your
system.

For now, the easiest way to install SpellvarDetection and its dependencies is
`pipenv <https://pipenv.readthedocs.io/en/latest/>`_. Simply run ``pipenv
install`` to install the required libraries. After the installation ``pipenv
shell`` starts a shell where SpellvarDetection is ready to be used.

Command line interface
----------------------

Here are some examples how to run a pipeline for detecting spelling variants
from the command line.

The commands often expect :ref:`input formatted as json<input>`, e.g. when
giving words to generate spelling variants for or for :ref:`describing a
spelling variant generation pipeline <pipeline>`. In this case, the commands
allow for either the json-formatted data as a string or the name of a file
containing the data.

The following command runs the union of the generators described in the
file ``example_data/simple_pipeline.json``:

.. code-block:: bash

   spellvardetection generate '["vnd"]' example_data/simple_pipeline.json --dictionary '["und", "vnde", "vnnde", "unde", "vns"]'

Some filters need to be trained, e.g. a `SVM
<https://scikit-learn.org/stable/modules/svm.html>`__ for distinguishing between
pairs of variants and non-variants using character n-grams from the aligned
words. Positive and negative examples are given in the files
``example_data/gml_positive_pairs`` and ``example_data/gml_negative_pairs``. The
trained model is written into the file ``example_data/gml_spellvar.model``:

.. code-block:: bash

    spellvardetection train filter '{"type": "sklearn", "options": {"classifier_clsname": "__svm__", "feature_extractors": [{"type": "surface"}]}}' example_data/gml_spellvar.model example_data/gml_positive_pairs example_data/gml_negative_pairs

The model can then be used to filter spelling variants:

.. code-block:: bash

   spellvardetection filter '{"vnd": ["und", "vns"]}' '{"type": "sklearn", "options": {"modelfile_name": "example_data/gml_spellvar.model"}}'

It can also be intergrated into a generator pipeline directly:

.. code-block:: bash

   spellvardetection generate '["vnd"]' example_data/svm_pipeline.json --dictionary '["und", "vnde", "vnnde", "unde", "vns"]'

The commands ``generate`` and ``filter`` both have the option ``-p`` that allows to
use multiple processes to work through a list of types in parallel. While this
can considerably speed up candidate generation and filtering, each process uses
its own copy of the used generators and filters, so this can use a lot of
main memory.

.. code-block:: bash

   spellvardetection generate '["vnd", "uns"]' example_data/svm_pipeline.json --dictionary '["und", "vnde", "vnnde", "unde", "vns"]' -p 2

The commands ``generate`` and ``filter`` both work on the type level, i.e. they
ignore the specific token context. To train and apply a token-based filter that
can distinguish different usages of a type, the following commands can be used
(here with the filter defined in ``example_data/token_filter.json`` trained on
the tokens given in ``example_data/gml_tokens.json``):

.. code-block:: bash

   spellvardetection train token_filter example_data/token_filter.json example_data/gml_spellvar_token.model example_data/gml_tokens.json '{"in": ["jn", "yn", "en", "ene"]}' --seed 42
   spellvardetection filter_tokens '[{"type": "in", "left_context": ["comet", "solen", "sic", "halden", "de"], "right_context": ["deme", "hove", "sint", ".", "De"], "variants": ["jn", "yn", "en"], "text": "Nowg._Schra_Rig.", "corpus": "ReN_1.0"}, {"type": "in", "left_context": ["doͮn", "id", "ne", "si", "dat"], "right_context": ["de", "paues", "sculdige", ".", "dat"], "variants": ["ene", "yn", "en"], "text": "Ssp._Berlin_Fragm._22", "corpus": "ReN_1.0"}]' '{"in": ["jn", "yn", "en", "ene"]}' '{"type": "cnn", "options": {"modelfile_name": "example_data/gml_spellvar_token.model"}}'


Web API
-------

Spelling variant detection pipelines can also be run using a Web API. Pipelines
and dictionaries that can be used with this API are stored in a database. There
is a simple command line interface to add them. To use ist, first set the
``FLASK_APP`` environment variable to ``spellvardetection.webapi``:

.. code-block:: bash

   export FLASK_APP=spellvardetection.webapi

Now you can add generators and dictionaries using the command line interface:

.. code-block:: bash

   flask db add-dictionary gml '["und", "vnde", "vnnde", "unde", "vns"]'
   flask db add-generator lev1 '{"type": "levenshtein", "options": {"max_dist": 1, "repetitions": "True"}}'

There also exists the command ``db clear`` to remove everything from the database.

To try it, a development server can be started with the following command:

.. code-block:: bash

   flask run

You can now try the API, e.g. using `curl <https://curl.haxx.se/>`__:
 
.. code-block:: bash

   curl http://127.0.0.1:5000/generate/lev1/gml/vnd

This will use the generator ``lev1`` to generate candidates for ``vnd`` from the
dictionary ``gml``.

Using post, candidates can be generated for multiple words:

.. code-block:: bash

   curl --header "Content-Type: application/json" --request POST --data '["vnd", "vnnd"]' http://127.0.0.1:5000/generate/lev1/gml

Resources like trained classifiers or brown clusters can be loaded from files.
Filenames can be given as absolute paths or -- easier -- as relative paths which
are relative to the resource folder in the apps instance folder.

.. warning::

   Resources might be loaded using pickle, which is not safe (see the `Python
   documentation <https://docs.python.org/3/library/pickle.html>`_). So make
   sure to include only files from trusted sources into your instance folder.

The resources in the instance folder can be managed using the command line interface:

.. code-block:: bash

    flask resources add example_data/gml.brown
    flask resources list

    flask db add-generator lev1brown '{"type": "pipeline", "options": {"generator": {"type": "levenshtein", "options": {"max_dist": 1, "repetitions": "True"}}, "type_filter": {"type": "cluster", "options": {"cluster_type": "brown", "cluster_file": "gml.brown"}}}}'
    curl http://127.0.0.1:5000/generate/lev1brown/gml/vnd
