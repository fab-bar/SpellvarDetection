import os
import json

from flask import request
import flask.json

from spellvardetection.rest.app import app

from spellvardetection.generator import createCandidateGenerator, _AbstractCandidateGenerator

def get_generator(generator, dictionary):

    try:
        generator_description = json.load(
            open(os.path.join(app.instance_path, 'pipeline', generator), 'r'))
    except OSError:
        return flask.json.jsonify(message='Could not load generator ' + generator), 400


    try:
        dict_ = json.load(
            open(os.path.join(app.instance_path, 'dict', dictionary), 'r'))
    except OSError:
        return flask.json.jsonify(message='Could not load dictionary ' + dictionary), 400


    try:
        generator_ = createCandidateGenerator(generator_description)
        generator_.setDictionary(dict_)

    except Exception:

        return flask.json.jsonify(message='Could not load the generator.'), 400

    return generator_


@app.route('/generate/<generator>/<dictionary>/<text_type>')
def generate_candidates(generator, dictionary, text_type):

    generator_ = get_generator(generator, dictionary)

    if not isinstance(generator_, _AbstractCandidateGenerator):
        return generator_
    else:
        return flask.json.dumps(list(generator_.getCandidatesForWord(text_type)))


@app.route('/generate/<generator>/<dictionary>', methods=['POST'])
def generate_candidates_for_multiple(generator, dictionary):

    generator_ = get_generator(generator, dictionary)
    if not request.is_json:
        return flask.json.jsonify(message='Types have to be given as json list.'), 400
    else:
        text_types = request.get_json()

    if not isinstance(generator_, _AbstractCandidateGenerator):
        return generator_
    else:
        return flask.json.dumps({word: list(variants) for word, variants in generator_.getCandidatesForWords(text_types).items()})
