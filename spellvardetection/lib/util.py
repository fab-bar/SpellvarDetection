# -*- coding: utf-8 -*-

import json
import statistics
import inspect

import numpy

def load_from_file_if_string(option):
    if isinstance(option, str):
        try:
            return json.loads(option)
        except:
            return json.load(open(option, 'r'))
    else:
        return option

## get all subclass
def all_subclasses(cls):
    return cls.__subclasses__() + [g for s in cls.__subclasses__()
                                   for g in all_subclasses(s)]

## create a factory for all subclasses of a given class
def create_factory(type_name, base_cls, create_func=None):

    factory_objects = dict()

    for cls in all_subclasses(base_cls):

        name = getattr(cls, 'name', None)
        if not inspect.isabstract(cls) and name is not None:
            if not (create_func is not None and hasattr(cls, create_func)):
                init_sig = inspect.signature(cls.__init__)
            else:
                init_sig = inspect.signature(getattr(cls, create_func))

            def object_initializer(cls):
                if create_func is not None and hasattr(cls, create_func):
                    return lambda options: getattr(cls, create_func)(**options)
                else:
                    return lambda options: cls(**options)

            factory_info = (
                ## get required arguments (those that have no default value)
                [param.name for param in init_sig.parameters.values() if param.name != 'self' and param.default is inspect.Parameter.empty],
                ## function to instantiate generator with given options
                object_initializer(cls)
            )

            factory_objects[name] = factory_info


    def factory(object_options):

        ## object_options are either a dict or encoded as json (possibly in a file)
        object_options = load_from_file_if_string(object_options)

        if 'type' not in object_options:
            raise ValueError('Type of object needs to be given')

        object_type = object_options['type']

        options = object_options.get('options', {})


        if object_type not in factory_objects:
            raise ValueError('No candidate ' + type_name + ' of type "' + object_type + '" exists.')

        missing_options = [name for name in factory_objects[object_type][0] if name not in options]
        if missing_options:
            raise ValueError('Missing options [' + ', '.join(missing_options) + '] for ' + type_name + ' of type ' + object_type)

        return factory_objects[object_type][1](options)

    return factory

### this function is used to allow to use an instance method in pool.map
### based on http://www.rueckstiess.net/research/snippets/show/ca1d7d90
def _unwrap_self(arg, function_name, **kwarg):
    return getattr(type(arg[0]), function_name)(*arg, **kwarg)

def nw_alignment(type_a, type_b):

    mismatch_cost = 1

    ### initalize cost matrix
    cost_matrix = numpy.zeros((len(type_a) + 1, len(type_b) + 1), dtype = int)

    for i in range(len(type_a) + 1):
        cost_matrix[i][0] = i*mismatch_cost

    for i in range(len(type_b) + 1):
        cost_matrix[0][i] = i*mismatch_cost

    for i in range(1, len(type_a) + 1):
        for j in range(1, len(type_b) + 1):
            if type_a[i-1] == type_b[j-1]:
                align_cost = 0
            else:
                align_cost = mismatch_cost
            cost_matrix[i][j] = min(cost_matrix[i-1][j-1] + align_cost, cost_matrix[i-1][j] + mismatch_cost, cost_matrix[i][j-1] + mismatch_cost)

    ### backtrack
    alignment = []
    i,j = len(type_a), len(type_b)
    while i > 0 and j > 0:
        if cost_matrix[i][j-1] == cost_matrix[i][j] - mismatch_cost:
            alignment.append(('-', type_b[j-1]))
            j -= 1
        elif cost_matrix[i-1][j] == cost_matrix[i][j] - mismatch_cost:
            alignment.append((type_a[i-1], '-'))
            i -= 1
        elif cost_matrix[i-1][j-1] == cost_matrix[i][j] or cost_matrix[i-1][j-1] == cost_matrix[i][j] - mismatch_cost:
            alignment.append((type_a[i-1], type_b[j-1]))
            i -= 1
            j -= 1
        else:
            raise Exception("NW-Alignment: No valid traceback path - Should not happen")

    while i > 0:
        alignment.append((type_a[i-1], '-'))
        i -= 1
    while j > 0:
        alignment.append(('-', type_b[j-1]))
        j -= 1

    return reversed(alignment)

def get_alignment(type_a, type_b, directed=False, conflate_id_pairs=False):

    seq_a = seq_b = []
    if not type_b: # handle empty candidate
        seq_a = type_a
        seq_b = "-"*len(type_a)
    else:
        ### result of needlemann-wunsch alignment is dependent on the order
        ### sort by length to avoid this
        if len(type_a) >= len(type_b):
            alignment = nw_alignment(type_a, type_b)
        else:
            alignment = nw_alignment(type_b, type_a)

    ## Convert alignment into sequence of aligned characters
    ## when directed=False, it is sorted
    ## when conflate_id_pairs=True, pairs like (f, f) are mapped to IDD
    if directed and conflate_id_pairs:
        return map(lambda x: u''.join(list(x)) if x[0] != x[1] else "IDD", alignment)
    elif conflate_id_pairs:
        return map(lambda x: u''.join(sorted(list(x))) if x[0] != x[1] else "IDD", alignemnt)
    elif directed:
        return map(lambda x: u''.join(list(x)), alignment)
    else:
        return map(lambda x: u''.join(sorted(list(x))), alignment)

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
