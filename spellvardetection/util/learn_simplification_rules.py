import difflib
import multiprocessing
import collections

from spellvardetection.generator import LevenshteinGenerator
from spellvardetection.lib.util import getTrueAndFalsePairs

def getRules(word1, word2, padding_left = "^^", padding_right = "^^"):

    word1 = padding_left + word1 + padding_right
    word2 = padding_left + word2 + padding_right

    opcodes = difflib.SequenceMatcher(None, word1, word2).get_opcodes()

    substitution_rules = [
        {
            'rule': tuple(sorted((word1[opcode[1]:opcode[2]], word2[opcode[3]:opcode[4]]), key=lambda x: (-len(x), x))),
            'left_context': word1[opcode[1]-1],
            'right_context': word1[opcode[2]]
        }
        for opcode in opcodes if opcode[0] in ['replace']]
    deletion_rules = [
        {
            'rule': tuple(sorted((word1[opcode[1]-1:opcode[2]], word2[opcode[3]-1:opcode[4]]), key=lambda x: -len(x))),
            'left_context': word1[opcode[1]-2],
            'right_context': word1[opcode[2]]
        }
        for opcode in opcodes if opcode[0] in ['delete', 'insert']]

    return tuple(substitution_rules + deletion_rules)

def getRulesFromPair(pair):

    return { rule['rule']: {'left_context': rule['left_context'], 'right_context': rule['right_context'], 'pair': set(pair) } for rule in getRules(pair[0], pair[1]) }


def getRulesAndFreqFromSpellvars(type_variants, max_processes=None):

    generator = LevenshteinGenerator(max_dist=1, merge_split=True)
    true_pairs, false_pairs = getTrueAndFalsePairs(type_variants, generator, max_processes)

    # Getting true rules
    with multiprocessing.Pool(max_processes) as pool:
        true_simplification_rules = pool.map(getRulesFromPair, true_pairs)

    # Getting all rules
    with multiprocessing.Pool(max_processes) as pool:
        false_simplification_rules = pool.map(getRulesFromPair, false_pairs)

    # Collect rules
    rules = {}
    for rule_feat in true_simplification_rules:
        for rule, _ in rule_feat.items():
            if rule not in rules:
                rules[rule] = {'tp': 0, 'fp': 0}
            rules[rule]['tp'] += 1

    for rule_feat in false_simplification_rules:
        for rule, _ in rule_feat.items():
            if rule in rules:
                rules[rule]['fp'] += 1

    return rules

