.. _input:

Preparing your data
===================

Data is passed as `json <https://json.org>`_ to the spelling variant detection
pipeline.

.. _json_note:
.. note::
   Whenever json is expected as input, the data can either be passed in
   directly or it can be written to a file. 
 

.. _dictionary:

Dictionary
----------

A dictionary is just a list containing types.

.. code-block:: json

   ["vnd", "und", "vns"]

.. _spellvar_dictionary:

Spelling variant dictionary
---------------------------

A spelling variant dictionary consists of types and a list of spelling variants
for each of these types. The ``generate`` and ``filter`` commands both output a
speling variant dictionary.

.. code-block:: json

   {"vnd": ["und", "vnde", "unde"], "und": ["unde", "vnd", "vnde"], "vns": ["uns"]}


Frequency dictionary
--------------------

A frequency dictionary consists of types and their frequency.

.. code-block:: json

   {"vnd": 42, "und": 201, "vns": 123}


Token list
----------

Token-based tasks (``token_filter`` and the util ``evaluate``) expect a list of
tokens as input. A token is a json object containing the ``type``, and lists of
types for each the ``left_context`` and the ``right_context``. Optionally, a
token can contain known ``variants`` (for training and evaluation),
``candidates`` from a type-based generater/filter and ``filtered_candidates``
from a token-based filter.

.. code-block:: json

    {"type": "und", "variants": ["vnd", "vnnde"], "candidates": ["vnd", "vns"], "filtered_candidates": ["vnd"]}
