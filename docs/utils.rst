Utils
=====

The commandline interface also contains different utils that can be useful for
experimenting with spelling variant detection.

Evaluation
----------

.. code-block:: bash

   spellvardetection utils evaluate '[{"type": "und", "variants": ["vnd", "vnnde"]}, {"type": "uns", "variants": ["vns"]}]' -p '{"und": ["vnd", "vns"], "uns": ["vns"]}'
   spellvardetection utils evaluate '[{"type": "und", "variants": ["vnd", "vnnde"], "filtered_candidates": ["vnd"]}, {"type": "uns", "variants": ["vns"], "filtered_candidates": ["vns"]}]'

This command prints (token-based) precision, recall and F1 together with the
mean number of predicted spelling variants and its standard deviation for
type-based spelling variant detection. As gold data it takes a list of tokens
with given *variants*. As prediction data it can use the *filtered_candidates*
included in the token list as output by a token-based filter. Alternatively, a
spelling variant dictionary as created by type-based generators and filter, i.e.
a dictionary that gives a list of predicted spelling variants for given types,
can be added with the option ``-p``.

Optionally, you can add a dictionary with ``-d``. This only consideres variants
and predicted variants from this dictionary for the evalulation. With ``-k``, you
can add a dictionary of known types that are ignored for the evaluation.

`Barteld et al. (2019) <https://doi.org/10.1007/s10579-018-09441-5>`_ introduces
two evaluation settings for spelling variant detection: text-eval and OOV-eval.
This is how to apply this two settings with the ``evaluate`` command given
training data in the files ``train_tokens``, ``train_types`` and test (or
development) data in the files ``test_tokens``, ``test_types``:

For text-eval, the aim is to produce spelling variants from the test data for
each token in the test data:

.. code-block:: bash

   spellvardetection generate test_types pipeline --dictionary test_types -o text_predictions
   spellvardetection utils evaluate test_tokens text_predictions -d test_types

For OOV-eval, the aim is to produce spelling variants from the training data for
each unknown (with respect to the training data) token in the test data:

.. code-block:: bash

   spellvardetection generate test_types pipeline --dictionary train_types -o oov_predictions
   spellvardetection utils evaluate test_tokens oov_predictions -d train_types -k train_types

.. _training_data:

Creating training data
----------------------

.. code-block:: bash

   spellvardetection utils extract_training_data spellvar_dict generated_spellvars positive_pairs negative_pairs

This command takes two :ref:`spelling variant dictionaries
<spellvar_dictionary>`: one containing actual spelling variants, the other
containing spelling variant candidates coming from a generator.

From these it creates two lists of word pairs: one containing pairs of spelling
variants, the other containing pairs that are not spelling variants (but that
are according to the generator). With this data a filter can be trained for the
used generator.

Feature extraction
------------------

.. code-block:: bash

   spellvardetection utils extract_features '[ {"type": "surface", "options": {"min_ngram_size": 2, "max_ngram_size": 4 }, "key": "ngrams"}]' example_data/gml_positive_pairs example_data/gml_negative_pairs -o gml_features

This command allows to run a list of feature extractors on a list of datapoints
and save it in the file given with ``-o``. This can be used to initialise the
cache of feature extractors as in the following example.

.. code-block:: bash

   spellvardetection train filter '{"type": "sklearn", "options": {"classifier_clsname": "__svm__", "feature_extractors": [{"type": "surface", "options": {"min_ngram_size": 2, "max_ngram_size": 4, "key": "ngrams" }}]}}' example_data/gml_spellvar.model example_data/gml_positive_pairs example_data/gml_negative_pairs -c gml_features

This is useful when training multiple models with the same features, e.g. when
doing hyperparameter optimization, to only extract the features once and try
different hyperparameter settings with these features.
