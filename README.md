# SpellvarDetection

A tool for detecting spelling variants in non-standard texts.

For more information see the following paper:

```
@InProceedings{E17-4002,
  author = 	"Barteld, Fabian",
  title = 	"Detecting spelling variants in non-standard texts",
  booktitle = 	"Proceedings of the Student Research Workshop at the 15th Conference of the European Chapter of the Association for Computational Linguistics",
  year = 	"2017",
  publisher = 	"Association for Computational Linguistics",
  pages = 	"11--22",
  location = 	"Valencia, Spain",
  url = 	"http://aclweb.org/anthology/E17-4002"
}
```

# Usage

Please note that this software package is still under development and the user
interface will likely change.

For now, the easiest way to install the dependencies and to run a pipeline for
spelling detection is [pipenv](https://pipenv.readthedocs.io/en/latest/). Simply
run `pipenv install` to install the required libraries (append `--dev` to
install dependencies for development as well).

## Command line interface

Here are some examples how to run a pipeline for detecting spelling variants
from the command line.

The following command runs the union of the generators described in `example_data/simple_pipeline.json`:

    pipenv run spellvardetection generate '["vnd"]' example_data/simple_pipeline.json --dictionary '["und", "vnde", "vnnde", "unde", "vns"]'

Some filters need to be trained, e.g. a SVM for distinguishing betwwen pairs
of variants and non-variants using character n-grams from the aligned words.
Positive and negative examples are given in `example_data/gml_positive_pairs`
and `example_data/gml_negative_pairs`. The trained model is written into the
file `example_data/gml_spellvar.model`.

    pipenv run spellvardetection train filter '{"type": "sklearn", "options": {"classifier_clsname": "__svm__", "feature_extractors": [{"type": "surface"}]}}' example_data/gml_spellvar.model example_data/gml_positive_pairs example_data/gml_negative_pairs

The model can then be used to filter spelling variants:

    pipenv run spellvardetection filter '{"vnd": ["und", "vns"]}' '{"type": "sklearn", "options": {"modelfile_name": "example_data/gml_spellvar.model"}}'

It can also be intergrated into a generator pipeline directly:

    pipenv run spellvardetection generate '["vnd"]' example_data/svm_pipeline.json --dictionary '["und", "vnde", "vnnde", "unde", "vns"]'

The commands `generate` and `filter` both have the option `-p` that allows to
use multiple processes to work through a list of types in parallel. While this
can considerably speed up candidate generation and filtering, each process uses
its own copy of the used generators and filters, so this can use a lot of
memory.

    pipenv run spellvardetection generate '["vnd", "uns"]' example_data/svm_pipeline.json --dictionary '["und", "vnde", "vnnde", "unde", "vns"]' -p 2

## REST API

Spelling variant detection pipelines can also be run using a REST API. Pipelines
and dictionaries that can be used with this API are stored in a database. There
is a simple command line interface to add them. To use ist, start a pipenv shell
by running `pipenv shell`. First set the `FLASK_APP` environment variable to
`spellvardetection.rest`:

    export FLASK_APP=spellvardetection.rest

Now you can add generators and dictionaries using the command line interface:

    flask db add-dictionary gml '["und", "vnde", "vnnde", "unde", "vns"]'
    flask db add-generator lev1 '{"type": "levenshtein", "options": {"max_dist": 1, "repetitions": "True"}}'

There also exists the command `db clear` to remove everything from the database.

To try it, a development server can be started with the following command:

    flask run

You can now try the API:

    curl http://127.0.0.1:5000/generate/lev1/gml/vnd

This will use the generator `lev1` to generate candidates for `vnd` from the
dictionary `gml`.

Using post, candidates can be generated for multiple words:

    curl --header "Content-Type: application/json" --request POST --data '["vnd", "vnnd"]' http://127.0.0.1:5000/generate/lev1/gml
