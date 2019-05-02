import functools
import json
import multiprocessing

import click

from .lib.util import load_from_file_if_string, evaluate
from .util.spellvarfactory import create_base_factory

@click.group()
@click.pass_context
def main(ctx):

    if ctx.obj is None:
        ctx.obj = {}

    ctx.obj['factory'] = create_base_factory()

@main.command()
@click.pass_context
@click.argument('vocabulary')
@click.argument('generator_settings')
@click.option('-d', '--dictionary')
@click.option('-o', '--output_file', type=click.File('w'))
@click.option('-p', '--max_processes', type=click.INT, default=1)
def generate(ctx, vocabulary, generator_settings, dictionary, output_file, max_processes):

    vocabulary = load_from_file_if_string(vocabulary)
    settings = load_from_file_if_string(generator_settings)
    generator = ctx.obj['factory'].create_from_name("generator", generator_settings)

    ## 0 or negative numbers for allowing as many processes as cores
    if max_processes < 1:
        max_processes = multiprocessing.cpu_count()

    generator.setMaxProcesses(max_processes)

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
@click.pass_context
@click.argument('candidates')
@click.argument('filter_settings')
@click.option('-o', '--output_file', type=click.File('w'))
@click.option('-p', '--max_processes', type=click.INT, default=1)
def filter_(ctx, candidates, filter_settings, output_file, max_processes):

    candidates = load_from_file_if_string(candidates)
    settings = load_from_file_if_string(filter_settings)
    cand_filter = ctx.obj['factory'].create_from_name("type_filter", filter_settings)

    ## 0 or negative numbers for allowing as many processes as cores
    if max_processes < 1:
        max_processes = multiprocessing.cpu_count()

    with multiprocessing.Pool(max_processes) as pool:

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
@click.pass_context
@click.argument('filter_settings')
@click.argument('modelfile_name')
@click.argument('positive_pairs')
@click.argument('negative_pairs')
@click.option('-c', '--feature_cache')
def train_filter(ctx, filter_settings, modelfile_name, positive_pairs, negative_pairs, feature_cache=None):

    positive_pairs = load_from_file_if_string(positive_pairs)
    negative_pairs = load_from_file_if_string(negative_pairs)

    filter_settings = load_from_file_if_string(filter_settings)

    ## if global feature cache is set - add this cache to all feature extractors
    if feature_cache is not None and 'feature_extractors' in filter_settings['options']:
        feature_cache = load_from_file_if_string(feature_cache)

        for feature_extractor in filter_settings['options']['feature_extractors']:
            if 'key' in feature_extractor.get('options', {}):
                feature_extractor['options']['cache'] = feature_cache

    cand_filter = ctx.obj['factory'].create_from_name("trainable_type_filter", filter_settings)
    cand_filter.train(positive_pairs, negative_pairs)
    cand_filter.save(modelfile_name)


@main.group()
def utils():
    pass


@utils.command('evaluate')
@click.argument('gold_file')
@click.argument('prediction_file')
@click.option('-d', '--dict_file')
@click.option('-k', '--known_file')
@click.option('-f', '--freq_file')
def evaluate_command(gold_file, prediction_file, dict_file=None, known_file=None, freq_file=None):

    tokens = load_from_file_if_string(gold_file)
    predictions_type = load_from_file_if_string(prediction_file)

    # add type-based predictions to tokens
    tokens = [{**token, **{'filtered_candidates': predictions_type.get(token['type'], [])}} for token in tokens]

    dictionary = set()
    if dict_file:
        dictionary = set(load_from_file_if_string(dict_file))

    known_dict = set()
    if known_file:
        known_dict = set(load_from_file_if_string(known_file))

    freq_dict = {}
    if freq_file:
        freq_dict = load_from_file_if_string(freq_file)

    print(
        "{:.2f}|{:.2f}|{:.2f}|{:.2f}+-{:.2f}".format(
            *evaluate(tokens, dictionary, known_dict, freq_dict)))
