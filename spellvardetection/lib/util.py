# -*- coding: utf-8 -*-

import statistics
import inspect

import jsonpickle
import numpy

def load_from_file_if_string(option):
    if isinstance(option, str):
        try:
            return jsonpickle.decode(option)
        except:
            with open(option, 'r') as jsonfile:
                jsonstring = jsonfile.read()
            return jsonpickle.decode(jsonstring)
    else:
        return option

### this function is used to allow to use an instance method in pool.map
### based on http://www.rueckstiess.net/research/snippets/show/ca1d7d90
def _unwrap_self(arg, function_name, **kwarg):
    return getattr(type(arg[0]), function_name)(*arg, **kwarg)

def nw_alignment(type_a, type_b, empty_char='-'):

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
            alignment.append((empty_char, type_b[j-1]))
            j -= 1
        elif cost_matrix[i-1][j] == cost_matrix[i][j] - mismatch_cost:
            alignment.append((type_a[i-1], empty_char))
            i -= 1
        elif cost_matrix[i-1][j-1] == cost_matrix[i][j] or cost_matrix[i-1][j-1] == cost_matrix[i][j] - mismatch_cost:
            alignment.append((type_a[i-1], type_b[j-1]))
            i -= 1
            j -= 1
        else:
            raise Exception("NW-Alignment: No valid traceback path - Should not happen")

    while i > 0:
        alignment.append((type_a[i-1], empty_char))
        i -= 1
    while j > 0:
        alignment.append((empty_char, type_b[j-1]))
        j -= 1

    return reversed(alignment)

def get_alignment(type_a, type_b, directed=False, conflate_id_pairs=False, empty_char='-'):

    seq_a = seq_b = []
    if not type_b: # handle empty candidate
        seq_a = type_a
        seq_b = empty_char*len(type_a)
    else:
        ### result of needlemann-wunsch alignment is dependent on the order
        ### sort by length to avoid this
        if len(type_a) >= len(type_b):
            alignment = nw_alignment(type_a, type_b, empty_char=empty_char)
        else:
            alignment = nw_alignment(type_b, type_a)

    ## Convert alignment into sequence of aligned characters
    ## when directed=False, it is sorted
    ## when conflate_id_pairs=True, pairs like (f, f) are mapped to IDD
    if directed and conflate_id_pairs:
        return map(lambda x: u''.join(list(x)) if x[0] != x[1] else "IDD", alignment)
    elif conflate_id_pairs:
        return map(lambda x: u''.join(sorted(list(x))) if x[0] != x[1] else "IDD", alignment)
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

    return [precision, recall, f1,
            statistics.mean(number_of_candidates),
            statistics.stdev(number_of_candidates)]

def getPairsFromSpellvardict(spellvardict):

    return set([
        tuple(sorted((word, spellvar))) for word, spellvars in spellvardict.items() for spellvar in spellvars
    ])

def getTrueAndFalsePairs(spellvardict, generator, max_processes=None):

    dictionary = set(spellvardict.keys())
    for word in spellvardict.keys():
        dictionary.update(spellvardict[word])

    ## extract all possible pairs
    generator.setDictionary(dictionary)
    generator.setMaxProcesses(max_processes)
    cand_pairs = getPairsFromSpellvardict(generator.getCandidatesForWords(dictionary))

    ## extract all positive pairs that would be generated
    true_pairs = getPairsFromSpellvardict(spellvardict).intersection(cand_pairs)

    ## get the false pairs
    false_pairs = cand_pairs.difference(true_pairs)

    return true_pairs, false_pairs

def get_positive_and_negative_pairs_with_context(tokens, spellvardict):

    positive_pairs = []
    negative_pairs = []

    for token in tokens:
        for candidate in spellvardict.get(token['type'], []):
            example = (
                token['type'], candidate, token['left_context'], token['right_context']
            )
            if candidate in token['variants']:
                positive_pairs.append(example)
            else:
                negative_pairs.append(example)

    return (positive_pairs, negative_pairs)

