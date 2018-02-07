# -*- coding: utf-8 -*-

import json
import statistics

def load_from_file_if_string(option):
    if isinstance(option, str):
        return json.load(open(option, 'r'))
    else:
        return option

def createFactory(factory_type, object_types):

    def factory(object_type, options):

        if object_type not in object_types:
            raise ValueError('No candidate ' + factory_type + ' of type "' + object_type + '" exists.')

        if not all(name in options for name in object_types[object_type][0]):
            raise ValueError('Missing options for ' + factory_type + ' of type ' + object_type)

        return object_types[object_type][1](options)

    return factory

### this function is used to allow to use an instance method in pool.map
### based on http://www.rueckstiess.net/research/snippets/show/ca1d7d90
def _unwrap_self(arg, function_name, **kwarg):
    return getattr(type(arg[0]), function_name)(*arg, **kwarg)


def evaluate(tokens, dictionary={}, known_dict={}, freq_dict={}):

    def isFrequent(toktext):
        return freq_dict.get(toktext, 0) > 9

    tp = 0
    fp = 0
    fn = 0

    number_of_candidates = []

    for token in tokens:
        type_text = token['type']

        ## skip known words
        if type_text in known_dict:
            continue

        ## skip frequent words
        if isFrequent(type_text):
            continue

        gold_variants = set(token['variants'])
        pred_variants = set(token['filtered_candidates'])

        if dictionary:
            gold_variants = gold_variants.intersection(dictionary)
            pred_variants = pred_variants.intersection(dictionary)

        ## filter frequent spelling variants
        gold_variants = set([variant for variant in gold_variants if not isFrequent(variant)])
        pred_variants = set([variant for variant in pred_variants if not isFrequent(variant)])

        tp += len(gold_variants.intersection(pred_variants))
        fp += len(pred_variants.difference(gold_variants))
        fn += len(gold_variants.difference(pred_variants))

        number_of_candidates.append(len(pred_variants))


    precision = tp/(tp + fp) if tp + fp > 0 else 1
    recall = tp/(tp + fn) if tp + fn > 0 else 1
    f1 = 2*(precision * recall)/(precision + recall) if precision + recall > 0 else 0

    return "|".join(
        map(lambda s: str("%.2f" % s),
            [
                precision,
                recall,
                f1
            ])) + "|" + str(
                "%.2f" % statistics.mean(number_of_candidates)) + '+-' + str(
                "%.2f" % statistics.stdev(number_of_candidates))
