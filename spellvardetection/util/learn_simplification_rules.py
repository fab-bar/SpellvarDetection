import difflib
import multiprocessing
import collections

from spellvardetection.generator import LevenshteinGenerator

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

    dictionary = set(type_variants.keys())
    for word in type_variants.keys():
        dictionary.update(type_variants[word])

    ## extract all possible pairs
    generator = LevenshteinGenerator(dictionary, 1, merge_split=True)
    generator.setMaxProcesses(max_processes)
    cand_pairs = getPairsFromSpellvardict(generator.getCandidatesForWords(dictionary))

    ## extract all positive pairs that would be generated
    true_pairs = getPairsFromSpellvardict(type_variants).intersection(cand_pairs)

    ## get the false pairs
    false_pairs = cand_pairs.difference(true_pairs)

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

def getPairsFromSpellvardict(spellvardict):

    return set([
        tuple(sorted((word, spellvar))) for word, spellvars in spellvardict.items() for spellvar in spellvars
    ])
