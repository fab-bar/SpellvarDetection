### Code is based on PySimpleAutomata (https://github.com/Oneiroe/PySimpleAutomata/)

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

def _epsilon_closure(states, epsilon, transitions):
    ## add epsilon closure
    new_states = states
    while new_states:
        curr_states = set()
        for state in new_states:
            curr_states.update(transitions.get(state, {}).get(epsilon, set()))

        new_states = curr_states - states
        states.update(curr_states)

def nfa_determinization(nfa: dict, any_input=None, epsilon=None) -> dict:
    dfa = {
        'initial_state': None,
        'accepting_states': set(),
        'transitions': dict()
    }

    initial_states =  nfa['initial_states']
    _epsilon_closure(initial_states, epsilon, nfa['transitions'])
    initial_states = frozenset(initial_states)
    dfa['initial_state'] = initial_states
    if initial_states.intersection(nfa['accepting_states']):
        dfa['accepting_states'].add(initial_states)

    sets_states = set()
    sets_queue = list()
    sets_queue.append(initial_states)
    sets_states.add(initial_states)

    while sets_queue:
        current_set = sets_queue.pop(0)
        new_transitions = {}
        for state in current_set:
            old_transitions = nfa['transitions'].get(state, {})
            for a in old_transitions.keys():
                if a != epsilon:
                    new_transitions[a] = new_transitions.get(a, set()) | old_transitions[a]

        for char, value in new_transitions.items():

            next_set = value | new_transitions.get(any_input, set())
            _epsilon_closure(next_set, epsilon, nfa['transitions'])

            next_set = frozenset(next_set)
            if next_set not in sets_states:
                sets_states.add(next_set)
                sets_queue.append(next_set)
                if next_set.intersection(nfa['accepting_states']):
                    dfa['accepting_states'].add(next_set)

            dfa['transitions'].setdefault(current_set, {})[char] = next_set

    return dfa


def dfa_intersection_language(dfa_1: dict, dfa_2: dict, any_input=None) -> dict:

    language = set()

    boundary = [(dfa_1['initial_state'], dfa_2['initial_state'])]
    while boundary:
        (state_dfa_1, state_dfa_2) = boundary.pop()
        if state_dfa_1 in dfa_1['accepting_states'] and state_dfa_2 in dfa_2['accepting_states']:
            language.add(state_dfa_2)

        if any_input in dfa_1['transitions'].get(state_dfa_1, {}):
            characters = dfa_2['transitions'].get(state_dfa_2, {}).keys()
        elif any_input in dfa_2['transitions'].get(state_dfa_2, {}):
            characters = dfa_1['transitions'].get(state_dfa_1, {}).keys()
        else:
            characters = set(dfa_1['transitions'].get(state_dfa_1, {}).keys()).intersection(dfa_2['transitions'].get(state_dfa_2, {}).keys())

        for a in characters:
            next_state_1 = dfa_1['transitions'][state_dfa_1].get(a, dfa_1['transitions'][state_dfa_1].get(any_input, frozenset()))
            next_state_2 = dfa_2['transitions'][state_dfa_2].get(a, dfa_2['transitions'][state_dfa_2].get(any_input, frozenset()))
            boundary.append((next_state_1, next_state_2))

    return language

