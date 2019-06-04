Getting started
===============

Please note that this software package is still under development and the user
interface will likely change.

For now, the easiest way to install the dependencies and to run a pipeline for
spelling detection is `pipenv <https://pipenv.readthedocs.io/en/latest/>`_. Simply
run ``pipenv install`` to install the required libraries (append ``--dev`` to
install dependencies for development as well). After the installation ``pipenv shell``
starts a shell where SpellvarDetection is ready to be used.

Command line interface
----------------------

Here are some examples how to run a pipeline for detecting spelling variants
from the command line.

The commands often expect input formatted as `json <https://json.org/>`_, e.g.
when giving words to generate spelling variants for or for describing a spelling
variant generation pipeline. In this case, the commands allow for either the
json-formatted data as a string or the name of a file containing the data.

The following command runs the union of the generators described in ``example_data/simple_pipeline.json``:
::

    spellvardetection generate '["vnd"]' example_data/simple_pipeline.json --dictionary '["und", "vnde", "vnnde", "unde", "vns"]'

Some filters need to be trained, e.g. a SVM for distinguishing betwwen pairs
of variants and non-variants using character n-grams from the aligned words.
Positive and negative examples are given in ``example_data/gml_positive_pairs``
and ``example_data/gml_negative_pairs``. The trained model is written into the
file ``example_data/gml_spellvar.model``.
::

    spellvardetection train filter '{"type": "sklearn", "options": {"classifier_clsname": "__svm__", "feature_extractors": [{"type": "surface"}]}}' example_data/gml_spellvar.model example_data/gml_positive_pairs example_data/gml_negative_pairs

The model can then be used to filter spelling variants:
::

    spellvardetection filter '{"vnd": ["und", "vns"]}' '{"type": "sklearn", "options": {"modelfile_name": "example_data/gml_spellvar.model"}}'

It can also be intergrated into a generator pipeline directly:
::

    spellvardetection generate '["vnd"]' example_data/svm_pipeline.json --dictionary '["und", "vnde", "vnnde", "unde", "vns"]'

The commands ``generate`` and ``filter`` both have the option ``-p`` that allows to
use multiple processes to work through a list of types in parallel. While this
can considerably speed up candidate generation and filtering, each process uses
its own copy of the used generators and filters, so this can use a lot of
main memory.
::

    spellvardetection generate '["vnd", "uns"]' example_data/svm_pipeline.json --dictionary '["und", "vnde", "vnnde", "unde", "vns"]' -p 2

Web API
-------

Spelling variant detection pipelines can also be run using a Web API. Pipelines
and dictionaries that can be used with this API are stored in a database. There
is a simple command line interface to add them. To use ist, first set the
``FLASK_APP`` environment variable to ``spellvardetection.webapi``:
::

    export FLASK_APP=spellvardetection.webapi

Now you can add generators and dictionaries using the command line interface:
::

    flask db add-dictionary gml '["und", "vnde", "vnnde", "unde", "vns"]'
    flask db add-generator lev1 '{"type": "levenshtein", "options": {"max_dist": 1, "repetitions": "True"}}'

There also exists the command ``db clear`` to remove everything from the database.

To try it, a development server can be started with the following command:
::

    flask run

You can now try the API:
::

    curl http://127.0.0.1:5000/generate/lev1/gml/vnd

This will use the generator ``lev1`` to generate candidates for ``vnd`` from the
dictionary ``gml``.

Using post, candidates can be generated for multiple words:
::

    curl --header "Content-Type: application/json" --request POST --data '["vnd", "vnnd"]' http://127.0.0.1:5000/generate/lev1/gml

Resources like trained classifiers or brown clusters can be loaded from files.
Filenames can be given as absolute paths or -- easier -- as relative paths which
are relative to the resource folder in the apps instance folder.
Warning: resources might be loaded using pickle, which is not safe. So make sure
to include only files from trusted sources into your instance folder.

The resources in the instance folder can be managed using the command line interface:
::

    flask resources add example_data/gml.brown
    flask resources list

    flask db add-generator lev1brown '{"type": "pipeline", "options": {"generator": {"type": "levenshtein", "options": {"max_dist": 1, "repetitions": "True"}}, "type_filter": {"type": "cluster", "options": {"cluster_type": "brown", "cluster_file": "gml.brown"}}}}'
    curl http://127.0.0.1:5000/generate/lev1brown/gml/vnd
