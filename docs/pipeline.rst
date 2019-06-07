.. _pipeline:

Creating a pipeline
===================

SpellvarDetection uses pipelines to generate spelling variants for a given
wordform. A pipeline contains at least a generator that creates spelling variant
candidates for the given wordform. Optionally, a filter can then be applied to
these candidates.

One example: when looking for spelling variants for ``und``, a generator might
return the candidates ``["vnd", "unde", "vnnd" "uns"]`` based on their surface
similarity, where only the first three are actual spelling variants of ``und``.
Applying a filter on this list can remove the false candidate ``uns``.

Pipelines are definied using :ref:`json <json_note>`:

.. code-block:: json

  {
    "type": "pipeline",
    "options": {
      "generator": "generator_definition",
      "type_filter": "filter_definition"
    }
  }

Recommended pipelines
---------------------

A good starting point is to use the Levenshtein generator to get candidates from
a :ref:`dictionary <dictionary>` within a given `edit distance
<https://en.wikipedia.org/wiki/Edit_distance>`_. The following definition uses a
variant of the Levenshtein distance that adds `transpositions
<https://en.wikipedia.org/wiki/Damerau%E2%80%93Levenshtein_distance>`_ as basic
edit operation and allows the repetition of characters withou assigning a cost:

.. code-block:: json

  {
    "type": "levenshtein",
    "options": {
      "max_dist": 1,
      "transposition": "True",
      "repetitions": "True"
    }
  }

If you have examples for spelling variants, you can train a filter that learns
differences between wordforms which are unlikely to appear in spelling variants
(for example *d* and *s* in ``und``, ``uns``).

For training a model two lists of word pairs are needed: a list of examples for
spelling variants and a list of pairs that are not spelling variants. The util
:ref:`create_training_data <training_data>` can create such lists from a
:ref:`spelling variant dictionary <spellvar_dictionary>` and the output from the
generator that should be used with the filter.

A trainable filter is trained with the command ``train filter``:

.. code-block:: bash

  spellvardetection train filter filter_definition.json modelfile positive_pairs negative_pairs

The trained filter is saved in modelfile.

The following filter definition trains a `BalancedBaggingClassifier
<https://imbalanced-learn.readthedocs.io/en/stable/ensemble.html#bagging>`_ to
filter spelling variant candidates based on differences like *d* and *s* in
``und``, ``uns``:

.. code-block:: json

  {
    "type": "sklearn",
    "options": {
      "classifier_clsname": "__svm__",
      "feature_extractors": [{"type": "surface"}]
    }
  }


To apply the trained filter in your pipeline use the following filter definition:
 
.. code-block:: json

  {
    "type": "sklearn",
    "options": {
      "modelfile_name": "modelfile"
    }
  }

When you have training data, it is also useful to add the known spelling
variants. This can be done with a ``union`` which combines the candidates
created by multiple generators. With the following definition, the results from
a pipeline are combined with known spelling variants:

.. code-block:: json

  {
    "type": "union",
    "options": {
      "generators": [
        "pipeline_definition.json",
        {
          "type": "lookup",
          "options": {
            "spellvar_dictionary": "spellvar_dictionary.json"
          }
        }
      ]
    }
  }

.. note::
   A pipeline is a specific type of a generator, therefore it can also be
   used as generator in a pipeline. Thereby it is possible to apply multiple
   filters.

A filter that is useful without training data is the ``cluster`` filter. This
filter removes candidates that are not in the same cluster as the given word. It
can be used with `Brown clusters
<https://en.wikipedia.org/wiki/Brown_clustering>`_ created with
https://github.com/percyliang/brown-cluster.
