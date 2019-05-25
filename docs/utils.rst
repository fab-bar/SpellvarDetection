Utils
=====

The commandline interface also contains different utils that can be useful for
experimenting with spelling variant detection.

Evaluation
----------
::

    spellvardetection utils evaluate '[{"type": "und", "variants": ["vnd", "vnnde"]}, {"type": "uns", "variants": ["vns"]}]' '{"und": ["vnd", "vns"], "uns": ["vns"]}'

This command prints (token-based) precision, recall and F1 together with the
mean number of predicted spelling variants and its standard deviation for
type-based spelling variant detection. As gold data it takes a list of tokens
with given variants. As prediction data it takes a spelling variant dictionary,
i.e. a dictionary that gives a list of predicted spelling variants for given
types.

Optionally, you can add a dictionary with ``-d``. This only consideres variants
and predicted variants from this dictionary for the evalulation. With ``-k``, you
can add a dictionary of known types that are ignored for the evaluation.

Feature extraction
------------------
::

    spellvardetection utils extract_features '[ {"type": "surface", "options": {"min_ngram_size": 2, "max_ngram_size": 4 }, "key": "ngrams"}]' example_data/gml_positive_pairs example_data/gml_negative_pairs -o gml_features

This command allows to run a list of feature extractors on a list of datapoints
and save it in the file given with ``-o``. This can be used to initialise the
cache of feature extractors as in the following example.
::

    spellvardetection train filter '{"type": "sklearn", "options": {"classifier_clsname": "__svm__", "feature_extractors": [{"type": "surface", "options": {"min_ngram_size": 2, "max_ngram_size": 4, "key": "ngrams" }}]}}' example_data/gml_spellvar.model example_data/gml_positive_pairs example_data/gml_negative_pairs -c gml_features

This is useful when training multiple models with the same features, e.g. when
doing hyperparameter optimization, to only extract the features once and try
different hyperparameter settings with these features.
