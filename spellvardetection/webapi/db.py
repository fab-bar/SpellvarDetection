import json
import os
import shutil

import click
from flask import g, current_app
from flask.cli import AppGroup
from tinymongo import TinyMongoClient

from spellvardetection.lib.util import load_from_file_if_string

def get_db():
    if 'db' not in g:
        connection = TinyMongoClient(current_app.config['DATABASE'])
        g.db = connection.spellvardetection

    return g.db

def close_db(e=None):
    db = g.pop('db', None)

### CLI for database management

db_cli = AppGroup('db', short_help='Administrate the database used by SpellvarDetection.')

@db_cli.command('clear')
def clear_db_command():
    """Clear the existing data."""

    try:
        shutil.rmtree(current_app.config['DATABASE'])
    except OSError:
        pass

    click.echo('Database cleared.')

@db_cli.command('add-dictionary')
@click.argument('name')
@click.argument('dictionary')
def add_dictionary_command(name, dictionary):
    """Add the given dicitonary."""

    db = get_db()

    if db.dictionaries.find_one({'name': name}) is not None:
        click.echo('There already exists a dictionary with the name ' + name + '.')
    else:
        db.dictionaries.insert_one(
            {
                "name": name,
                "dict": load_from_file_if_string(dictionary)
            }
        )

@db_cli.command('add-generator')
@click.argument('name')
@click.argument('generator')
def add_generator_command(name, generator):
    """Add the given generator."""

    db = get_db()

    if db.generators.find_one({'name': name}) is not None:
        click.echo('There already exists a generator with the name ' + name + '.')
    else:
        db.generators.insert_one(
            {
                "name": name,
                **load_from_file_if_string(generator)
            }
        )


def init_app(app):
    app.teardown_appcontext(close_db)
    app.cli.add_command(db_cli)
