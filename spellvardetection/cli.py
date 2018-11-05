import json

import click

from .generator import createCandidateGenerator
from .lib.util import load_from_file_if_string

@click.group()
def main():
    pass

@main.command()
@click.argument('vocabulary')
@click.argument('generator_settings')
@click.option('-d', '--dictionary')
@click.option('-o', '--output_file', type=click.File('w'))
def generate(vocabulary, generator_settings, dictionary, output_file):

    vocabulary = load_from_file_if_string(vocabulary)
    settings = load_from_file_if_string(generator_settings)
    generator = createCandidateGenerator(settings)

    if dictionary:
        generator.setDictionary(dictionary)

    try:
        variants = generator.getCandidatesForWords(vocabulary)
    except Exception as e:
        print(e)
    else:
        click.echo(
            json.dumps({word: list(variants) for word, variants in variants.items()}),
            file=output_file)

