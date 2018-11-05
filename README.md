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

    pipenv run ./bin/train_filter '{"type": "sklearn", "options": {"classifier_clsname": "__svm__", "feature_extractors": [{"type": "surface", "name": "surface"}]}}' example_data/gml_spellvar.model example_data/gml_positive_pairs example_data/gml_negative_pairs

The model can then be used to filter spelling variants:

    pipenv run ./bin/filter '{"vnd": ["und", "vns"]}' '{"type": "sklearn", "options": {"modelfile_name": "example_data/gml_spellvar.model"}}'

It can also be intergrated into a generator pipeline directly:

    pipenv run spellvardetection generate '["vnd"]' example_data/svm_pipeline.json --dictionary '["und", "vnde", "vnnde", "unde", "vns"]'


## REST API

Spelling variant detection pipelines can also be run using a REST API.
To try it, a development server can be started with the following command:

    pipenv run bash -c 'export FLASK_APP=spellvardetection.rest; flask run'


To use pipelines and dictionaries they have to be stored in the subfolders
`pipeline` and `dict` of an [instance
folder](http://flask.pocoo.org/docs/1.0/config/#instance-folders):

    curl http://127.0.0.1:5000/generate/generator/dictionary/vnd

This will use the generator described in the file `pipeline/generator_file` to
generate candidates for `vnd` from the dictionary in the file `dict/dictionary`.

Using post, candidates can be generated for multiple words:

    curl --header "Content-Type: application/json" --request POST --data '["vnd", "vnnd"]' http://127.0.0.1:5000/generate/generator/dictionary
