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

::

   ["vnd", "und", "vns"]

.. _spellvar_dictionary:

Spelling variant dictionary
---------------------------

A spelling variant dictionary consists of types and a list of spelling variants
for each of these types. The ``generate`` and ``filter`` commands both output a
speling variant dictionary.

::

   {"vnd": ["und", "vnde", "unde"], "und": ["unde", "vnd", "vnde"], "vns": ["uns"]}


Frequency dictionary
--------------------

A frequency dictionary consists of types and their frequency.

::

   {"vnd": 42, "und": 201, "vns": 123}
