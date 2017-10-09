### Code from PySimpleAutomata (https://github.com/Oneiroe/PySimpleAutomata/)

# MIT License

# Copyright (c) 2017 Alessio Cecconi

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

### nfa_determinization changed:
###    use hash of (frozen)set instead of str to represent states
###    the string-representation of sets is not deterministic
import warnings

def nfa_determinization(nfa: dict) -> dict:
    """ Returns a DFA that reads the same language of the input NFA.
    Let A be an NFA, then there exists a DFA :math:`A_d` such
    that :math:`L(A_d) = L(A)`. Intuitively, :math:`A_d`
    collapses all possible runs of A on a given input word into
    one run over a larger state set.
    :math:`A_d` is defined as:
    :math:`A_d = (Σ, 2^S , s_0 , ρ_d , F_d )`
    where:
    • :math:`2^S` , i.e., the state set of :math:`A_d` , consists
      of all sets of states S in A;
    • :math:`s_0 = S^0` , i.e., the single initial state of
      :math:`A_d` is the set :math:`S_0` of initial states of A;
    • :math:`F_d = \{Q | Q ∩ F ≠ ∅\}`, i.e., the collection of
      sets of states that intersect F nontrivially;
    • :math:`ρ_d(Q, a) = \{s' | (s,a, s' ) ∈ ρ\ for\ some\ s ∈ Q\}`.
    :param dict nfa: input NFA.
    :return: *(dict)* representing a DFA
    """
    dfa = {
        'alphabet': nfa['alphabet'].copy(),
        'initial_state': None,
        'states': set(),
        'accepting_states': set(),
        'transitions': dict()
    }

    initial_states = frozenset(nfa['initial_states'])
    initial_state = hash(initial_states)
    if len(nfa['initial_states']) > 0:
        dfa['initial_state'] = initial_state
        dfa['states'].add(initial_state)

    sets_states = list()
    sets_queue = list()
    sets_queue.append(initial_states)
    sets_states.append(initial_states)
    if len(nfa['initial_states'].intersection(nfa['accepting_states'])) > 0:
        dfa['accepting_states'].add(initial_state)

    while sets_queue:
        current_set = sets_queue.pop(0)
        for a in dfa['alphabet']:
            next_set = set()
            for state in current_set:
                if (state, a) in nfa['transitions']:
                    for next_state in nfa['transitions'][state, a]:
                        next_set.add(next_state)
            if len(next_set) == 0:
                continue
            next_set = frozenset(next_set)
            if next_set not in sets_states:
                sets_states.append(next_set)
                sets_queue.append(next_set)
                set_id = hash(next_set)
                if set_id in dfa['states']:
                    warn('Hash collision in nfa determinization', warnings.RuntimeWarning)
                else:
                    dfa['states'].add(set_id)
                if next_set.intersection(nfa['accepting_states']):
                    dfa['accepting_states'].add(set_id)

            dfa['transitions'][hash(current_set), a] = hash(next_set)

    return dfa


def dfa_intersection(dfa_1: dict, dfa_2: dict) -> dict:
    """ Returns a DFA accepting the intersection of the DFAs in
    input.
    Let :math:`A_1 = (Σ, S_1 , s_{01} , ρ_1 , F_1 )` and
    :math:`A_2 = (Σ, S_2 , s_{02} , ρ_2 , F_2 )` be two DFAs.
    Then there is a DFA :math:`A_∧` that runs simultaneously both
    :math:`A_1` and :math:`A_2` on the input word and
    accepts when both accept.
    It is defined as:
    :math:`A_∧ = (Σ, S_1 × S_2 , (s_{01} , s_{02} ), ρ, F_1 × F_2 )`
    where
    :math:`ρ((s_1 , s_2 ), a) = (s_{X1} , s_{X2} )` iff
    :math:`s_{X1} = ρ_1 (s_1 , a)` and :math:`s_{X2}= ρ_2 (s_2 , a)`
    Implementation proposed guarantees the resulting DFA has only
    **reachable** states.
    :param dict dfa_1: first input DFA;
    :param dict dfa_2: second input DFA.
    :return: *(dict)* representing the intersected DFA.
    """
    intersection = {
        'alphabet': dfa_1['alphabet'].intersection(dfa_2['alphabet']),
        'states': {(dfa_1['initial_state'], dfa_2['initial_state'])},
        'initial_state': (dfa_1['initial_state'], dfa_2['initial_state']),
        'accepting_states': set(),
        'transitions': dict()
    }

    boundary = set()
    boundary.add(intersection['initial_state'])
    while boundary:
        (state_dfa_1, state_dfa_2) = boundary.pop()
        if state_dfa_1 in dfa_1['accepting_states'] \
                and state_dfa_2 in dfa_2['accepting_states']:
            intersection['accepting_states'].add((state_dfa_1, state_dfa_2))

        for a in intersection['alphabet']:
            if (state_dfa_1, a) in dfa_1['transitions'] \
                    and (state_dfa_2, a) in dfa_2['transitions']:
                next_state_1 = dfa_1['transitions'][state_dfa_1, a]
                next_state_2 = dfa_2['transitions'][state_dfa_2, a]
                if (next_state_1, next_state_2) not in intersection['states']:
                    intersection['states'].add((next_state_1, next_state_2))
                    boundary.add((next_state_1, next_state_2))
                intersection['transitions'][(state_dfa_1, state_dfa_2), a] = \
                    (next_state_1, next_state_2)

    return intersection

