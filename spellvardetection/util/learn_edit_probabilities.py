from collections import defaultdict

from spellvardetection.generator import LevenshteinGenerator
from spellvardetection.lib.util import get_alignment, getTrueAndFalsePairs

def getProbabilitiesFromSpellvars(type_variants, max_processes=None):

    generator = LevenshteinGenerator(max_dist=1, transposition=True)
    true_pairs, false_pairs = getTrueAndFalsePairs(type_variants, generator, max_processes)

    ## initialize counts with 1 (Laplace smoothing)
    probabilities = defaultdict(lambda: {'correct': 1, 'false': 1})

    def get_edit_ops(pair):
        return filter(lambda e: e != 'IDD',
                      get_alignment(pair[0], pair[1],
                                    directed=False, conflate_id_pairs=True, empty_char=''))

    for pair in true_pairs:
        for edit_op in get_edit_ops(pair):
            probabilities[edit_op]['correct'] += 1

    for pair in false_pairs:
        for edit_op in get_edit_ops(pair):
            probabilities[edit_op]['false'] += 1

    return [{'char1': edit_op[0], 'char2': edit_op[1] if len(edit_op) == 2 else '',
             'probability': counts['correct']/(counts['correct'] + counts['false'])}
            for edit_op, counts in probabilities.items()]
