import atexit
import cProfile
import functools
import json
import multiprocessing
import random

import click
import jsonpickle

from .lib.util import load_from_file_if_string, evaluate, getPairsFromSpellvardict, get_positive_and_negative_pairs_with_context
from .util.spellvarfactory import create_base_factory
import spellvardetection.util.learn_simplification_rules
import spellvardetection.util.learn_edit_probabilities

class JsonOption(click.ParamType):
    """The json-option type allows for passing a list or dict using json as
    parameter. If the passed string is not valid json, it is interpreted as
    a filename and the content of the file is used.
    """

    name = 'json-option'

    def convert(self, value, param, ctx):
        try:
            result = load_from_file_if_string(value)
        except Exception:
            self.fail(
                value + " could not be parsed.",
                param,
                ctx,
            )

        return result


@click.group()
@click.option('--with_profiler', default=False, is_flag=True)
@click.pass_context
def main(ctx, with_profiler):

    if with_profiler:
        cp = cProfile.Profile()
        cp.enable()

        def stop_profiling():
            cp.disable()
            cp.print_stats(sort='time')

        atexit.register(stop_profiling)

    if ctx.obj is None:
        ctx.obj = {}

    ctx.obj['factory'] = create_base_factory()

@main.command()
@click.pass_context
@click.argument('vocabulary', type=JsonOption())
@click.argument('generator_settings', type=JsonOption())
@click.option('-d', '--dictionary', type=JsonOption())
@click.option('-o', '--output_file', type=click.File('w'))
@click.option('-p', '--max_processes', type=click.INT, default=1)
def generate(ctx, vocabulary, generator_settings, dictionary, output_file, max_processes):

    generator = ctx.obj['factory'].create_from_name("generator", generator_settings)

    ## 0 or negative numbers for allowing as many processes as cores
    if max_processes < 1:
        max_processes = multiprocessing.cpu_count()

    generator.setMaxProcesses(max_processes)

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

## Helper function for filter
def apply_filter(word_candidates, cand_filter):
    return (word_candidates[0], list(cand_filter.filterCandidates(word_candidates[0], word_candidates[1])))

@main.command('filter')
@click.pass_context
@click.argument('candidates', type=JsonOption())
@click.argument('filter_settings', type=JsonOption())
@click.option('-o', '--output_file', type=click.File('w'))
@click.option('-p', '--max_processes', type=click.INT, default=1)
def filter_(ctx, candidates, filter_settings, output_file, max_processes):

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


@main.command('filter_tokens')
@click.pass_context
@click.argument('tokens', type=JsonOption())
@click.argument('spellvarcandidates', type=JsonOption())
@click.argument('filter_settings', type=JsonOption())
@click.option('-o', '--output_file', type=click.File('w'))
def filter_tokens(ctx, tokens, spellvarcandidates, filter_settings, output_file):


    cand_filter = ctx.obj['factory'].create_from_name("token_filter", filter_settings)

    def filter_variants_for_token(token_candidates):
        token_candidates[0]['variants'] = cand_filter.filterCandidates(token_candidates[0]['type'], token_candidates[1], token_candidates[0]['left_context'], token_candidates[0]['left_context'])
        return token_candidates[0]

    filtered = [
        {
            **token,
            **{
                'candidates': spellvarcandidates.get(token['type'], []),
                'filtered_candidates': list(cand_filter.filterCandidates(token['type'], spellvarcandidates.get(token['type'], []), token['left_context'], token['right_context']))
            }
        }
        for token in tokens
    ]

    click.echo(
        json.dumps(filtered),
        file=output_file)


@main.group()
def train():
    pass

@train.command('filter')
@click.pass_context
@click.argument('filter_settings', type=JsonOption())
@click.argument('modelfile_name')
@click.argument('positive_pairs', type=JsonOption())
@click.argument('negative_pairs', type=JsonOption())
@click.option('-c', '--feature_cache', type=JsonOption())
def train_filter(ctx, filter_settings, modelfile_name, positive_pairs, negative_pairs, feature_cache=None):

    ## if global feature cache is set - add this cache to all feature extractors
    if feature_cache is not None and 'feature_extractors' in filter_settings['options']:

        for feature_extractor in filter_settings['options']['feature_extractors']:
            if 'key' in feature_extractor.get('options', {}):
                feature_extractor['options']['cache'] = feature_cache

    cand_filter = ctx.obj['factory'].create_from_name("trainable_type_filter", filter_settings)
    cand_filter.train(positive_pairs, negative_pairs)
    cand_filter.save(modelfile_name)


@train.command('token_filter')
@click.pass_context
@click.argument('filter_settings', type=JsonOption())
@click.argument('modelfile_name')
@click.argument('annotated_tokens', type=JsonOption())
@click.argument('spellvarcandidates', type=JsonOption())
def train_token_filter(ctx, filter_settings, modelfile_name, annotated_tokens, spellvarcandidates):

    cand_filter = ctx.obj['factory'].create_from_name("trainable_token_filter", filter_settings)

    ### get positive and negative pairs from tokens
    positive_pairs, negative_pairs = get_positive_and_negative_pairs_with_context(
        annotated_tokens, spellvarcandidates)

    cand_filter.train(positive_pairs, negative_pairs)
    cand_filter.save(modelfile_name)

@main.group()
def utils():
    pass

@utils.command('extract_training_data')
@click.argument('gold_variants', type=JsonOption())
@click.argument('generated_variants', type=JsonOption())
@click.argument('outfile_positive', type=click.File('w'))
@click.argument('outfile_negative', type=click.File('w'))
@click.option('-f', '--frequency', type=(JsonOption(), int), default=(None, 1))
@click.option('-r', '--resample', default=False, is_flag=True)
def extract_training_data(gold_variants, generated_variants,
                          outfile_positive, outfile_negative,
                          frequency, resample):

    freq_dict, freq_thresh = frequency

    pos_pairs = set([])
    neg_pairs = set([])

    spellvarpairs = getPairsFromSpellvardict(gold_variants)
    candidatepairs = getPairsFromSpellvardict(generated_variants)

    for pair in candidatepairs:
        if pair in spellvarpairs:
            pos_pairs.add(pair)
        else:
            neg_pairs.add(pair)

    if freq_dict is not None:
        neg_pairs = set(
            [neg_pair for neg_pair in neg_pairs
             if freq_dict.get(neg_pair[0], 0) >= freq_thresh and freq_dict.get(neg_pair[1], 0) >= freq_thresh])

    if resample:
        if len(neg_pairs) > len(pos_pairs):
            neg_pairs = random.sample(neg_pairs, len(pos_pairs))
        elif len(pos_pairs) > len(neg_pairs):
            pos_pairs = random.sample(pos_pairs, len(neg_pairs))

    json.dump(list(pos_pairs), outfile_positive)
    json.dump(list(neg_pairs), outfile_negative)

@utils.command('evaluate')
@click.argument('gold_data', type=JsonOption())
@click.option('-p', '--predictions', type=JsonOption())
@click.option('-d', '--dictionary', type=JsonOption())
@click.option('-k', '--known_dictionary', type=JsonOption())
@click.option('-f', '--freq_dict', type=JsonOption())
def evaluate_command(gold_data, predictions=None, dictionary=None, known_dictionary=None, freq_dict=None):

    # add type-based predictions to tokens
    if predictions is not None:
        tokens = [{**token, **{'filtered_candidates': predictions.get(token['type'], [])}} for token in gold_data]
    else:
        tokens = gold_data

    if dictionary is not None:
        dictionary = set(dictionary)
    else:
        dictionary = set()

    if known_dictionary is not None:
        known_dictionary = set(known_dictionary)
    else:
        known_dictionary = set()

    if freq_dict is None:
        freq_dict = {}

    print(
        "{:.2f}|{:.2f}|{:.2f}|{:.2f}+-{:.2f}".format(
            *evaluate(tokens, dictionary, known_dictionary, freq_dict)))


@utils.command('filter_similarity')
@click.argument('predictions', type=JsonOption())
@click.argument('sim_thresh', type=float)
@click.option('-o', '--output_file', type=click.File('w'))
def filter_similarity_command(predictions, sim_thresh, output_file):

    types = {type_: [variant for variant, sim in variants if sim >= sim_thresh]
        for type_, variants in
        predictions.items()}

    click.echo(json.dumps(types), file=output_file)


@utils.command('extract_features')
@click.argument('feature_extractors', type=JsonOption())
@click.argument('datapoints', nargs=-1, required=True, type=JsonOption())
@click.option('-o', '--output_file', type=click.File('w'))
@click.pass_context
def extract_features(ctx, feature_extractors, datapoints, output_file):

    datapoints = [datapoint for dps in datapoints for datapoint in dps]

    feature_cache = dict()

    for extractor in feature_extractors:

        extractor_key = extractor['key']

        feature_extractor = ctx.obj['factory'].create_from_name("extractor", extractor)
        feature_extractor.setFeatureCache(feature_cache, key=extractor_key)
        feature_extractor.extractFeatures(datapoints)

    ## using jsonpickle to allow datatypes not supported by json (e.g. set)
    click.echo(jsonpickle.encode(feature_cache), file=output_file)


@utils.command('simplify')
@click.pass_context
@click.argument('input_file', type=click.File('r'))
@click.argument('rules', type=JsonOption())
@click.option('-o', '--output_file', type=click.File('w'))
def evaluate_command(ctx, input_file, rules, output_file):

    simpl = ctx.obj['factory'].create_from_name("generator",
                                                ({'type': 'simplification', 'options': {'ruleset': rules, 'dictionary': []}}))

    for line in input_file:
        click.echo(
            simpl.getSimplification(line.strip()),
            file=output_file)


@utils.group()
def learn():
    pass

@learn.command('simplification_rules')
@click.argument('spellvar_dictionary', type=JsonOption())
@click.option('-f', '--freq_thresh', type=click.INT, default=30)
@click.option('-pr', '--prec_thresh', type=float, default=0.7)
@click.option('-p', '--max_processes', type=click.INT, default=1)
@click.option('-o', '--output_file', type=click.File('w'))
def learn_simplification_rules(spellvar_dictionary, freq_thresh, prec_thresh, max_processes, output_file):

    ## 0 or negative numbers for allowing as many processes as cores
    if max_processes < 1:
        max_processes = multiprocessing.cpu_count()

    rules = spellvardetection.util.learn_simplification_rules.getRulesAndFreqFromSpellvars(
        spellvar_dictionary, max_processes)

    click.echo(
        json.dumps([
            rule for rule, rule_features in rules.items()
            if rule_features['tp'] >= freq_thresh and rule_features['tp']/(rule_features['tp']+rule_features['fp']) >= prec_thresh]),
        file=output_file)

@learn.command('edit_probabilities')
@click.argument('spellvar_dictionary', type=JsonOption())
@click.option('-p', '--max_processes', type=click.INT, default=1)
@click.option('-o', '--output_file', type=click.File('w'))
def learn_edit_probabilities(spellvar_dictionary, max_processes, output_file):

    ## 0 or negative numbers for allowing as many processes as cores
    if max_processes < 1:
        max_processes = multiprocessing.cpu_count()

    probabilities = spellvardetection.util.learn_edit_probabilities.getProbabilitiesFromSpellvars(
        spellvar_dictionary, max_processes)

    click.echo(json.dumps(probabilities), file=output_file)
