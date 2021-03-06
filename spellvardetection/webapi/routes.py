import os
import json

from flask import Blueprint, current_app, request

import flask.json

from spellvardetection.webapi.db import get_db

from spellvardetection.generator import _AbstractCandidateGenerator

app_blueprint = Blueprint('app', __name__)

def get_generator(generator, dictionary):

    db = get_db()

    generator_description = db.generators.find_one({'name': generator})
    if generator_description is None:
        return flask.json.jsonify(message='Could not load generator ' + generator), 400

    dict_ = db.dictionaries.find_one({'name': dictionary})
    if dict_ is None:
        return flask.json.jsonify(message='Could not load dictionary ' + dictionary), 400

    try:
        generator_ = current_app.config['FACTORY'].create_from_name("generator", generator_description)
        generator_.setDictionary(dict_['dict'])

    except Exception:

        return flask.json.jsonify(message='Could not create the generator.'), 400

    return generator_


@app_blueprint.route('/generate/<generator>/<dictionary>/<text_type>')
def generate_candidates(generator, dictionary, text_type):

    generator_ = get_generator(generator, dictionary)

    if not isinstance(generator_, _AbstractCandidateGenerator):
        return generator_
    else:
        return flask.json.dumps(list(generator_.getCandidatesForWord(text_type)))


@app_blueprint.route('/generate/<generator>/<dictionary>', methods=['POST'])
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

