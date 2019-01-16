import functools
import json
import multiprocessing

import click

from .generator import createCandidateGenerator
from .type_filter import createTypeFilter, createTrainableTypeFilter
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
        generator.setDictionary(load_from_file_if_string(dictionary))

    try:
        variants = generator.getCandidatesForWords(vocabulary)
    except Exception as e:
        print(e)
    else:
        click.echo(
            json.dumps({word: list(variants) for word, variants in variants.items()}),
            file=output_file)

## Helper function for filter
def apply_filter(word_candidates, cand_filter):
    return (word_candidates[0], list(cand_filter.filterCandidates(word_candidates[0], word_candidates[1])))

@main.command('filter')
@click.argument('candidates')
@click.argument('filter_settings')
@click.option('-o', '--output_file', type=click.File('w'))
def filter_(candidates, filter_settings, output_file):

    candidates = load_from_file_if_string(candidates)
    settings = load_from_file_if_string(filter_settings)
    cand_filter = createTypeFilter(settings)


    with multiprocessing.Pool(multiprocessing.cpu_count()) as pool:

        filtered = pool.map(
            functools.partial(apply_filter, cand_filter=cand_filter),
            candidates.items())

    click.echo(
        json.dumps({word_variants[0]: word_variants[1] for word_variants in filtered}),
        file=output_file)

@main.group()
def train():
    pass

@train.command('filter')
@click.argument('filter_settings')
@click.argument('modelfile_name')
@click.argument('positive_pairs')
@click.argument('negative_pairs')
@click.option('-c', '--feature_cache')
def train_filter(filter_settings, modelfile_name, positive_pairs, negative_pairs, feature_cache=None):

    positive_pairs = load_from_file_if_string(positive_pairs)
    negative_pairs = load_from_file_if_string(negative_pairs)

    filter_settings = load_from_file_if_string(filter_settings)

    if feature_cache is not None and 'feature_extractors' in filter_settings['options']:
        feature_cache = load_from_file_if_string(feature_cache)

    for feature_extractor in filter_settings['options'].get('feature_extractors', []):
        if 'cache' in feature_extractor:
            feature_extractor['cache'] = load_from_file_if_string(feature_extractor['cache'])
        elif feature_cache is not None and 'key' in feature_extractor:
            feature_extractor['cache'] = feature_cache

    cand_filter = createTrainableTypeFilter(filter_settings)
    cand_filter.train(positive_pairs, negative_pairs)
    cand_filter.save(modelfile_name)
