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

## Examples

Here are some examples how to run a pipeline for detecting spelling variants.

The following command runs the union of the generators described in `example_data/simple_pipeline.json`:

    pipenv run ./bin/generate_candidates '["vnd"]' union example_data/simple_pipeline.json --dictionary '["und", "vnde", "vnnde", "unde", "vns"]'

Some filters need to be trained, e.g. a SVM for distinguishing betwwen pairs
of variants and non-variants using character n-grams from the aligned words.
Positive and negative examples are given in `example_data/gml_positive_pairs`
and `example_data/gml_negative_pairs`. The trained model is written into the
file `example_data/gml_spellvar.model`.

    pipenv run ./bin/train_filter sklearn '{"classifier_clsname": "__svm__", "feature_extractors": [{"type": "surface", "name": "surface"}]}' example_data/gml_spellvar.model example_data/gml_positive_pairs example_data/gml_negative_pairs

The model can then be used to filter spelling variants:

    pipenv run ./bin/filter '{"vnd": ["und", "vns"]}' sklearn '{"modelfile_name": "example_data/gml_spellvar.model"}'

It can also be intergrated into a generator pipeline directly:

    pipenv run ./bin/generate_candidates '["vnd"]' union example_data/svm_pipeline.json --dictionary '["und", "vnde", "vnnde", "unde", "vns"]'

