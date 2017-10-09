from spellvardetection.lib.simple_automata import dfa_intersection, nfa_determinization

class DictAutomaton:

    ANY_INPUT = '__ANY__'
    EPSILON = '__EPSILON__'

    def fuzzySearch(self, word, distance, merge_split=False, transposition=False, repetitions=False):

        lev_aut = self._create_levenshtein_dfa(word, distance, merge_split=merge_split, transposition=transposition, repetitions=repetitions)
        acceptor = dfa_intersection(lev_aut, self.automaton)
        return set([state[1] for state in acceptor['accepting_states']])

    def __init__(self, dictionary):

        ## create acceptor for dictionary

        initial_state = '_START_'

        states = {initial_state}
        input_symbols = set()
        final_states = set()
        transitions = dict()

        for word in dictionary:

            final_states.add(word)
            prefix = ''

            for character in word:

                input_symbols.add(character)

                if not prefix:
                    from_state = initial_state
                else:
                    from_state = prefix

                prefix = prefix + character

                states.add(prefix)

                transitions[(from_state, character)] = prefix


        self.automaton = {
            'alphabet': input_symbols,
            'states': states,
            'initial_state': initial_state,
            'accepting_states': final_states,
            'transitions': transitions
        }

    def _get_alphabet(self):

        return set(self.automaton['alphabet'])

    def _add_transition(self, transitions, source_state, character, target_states):

        if (source_state, character) not in transitions:
            transitions[(source_state, character)] = set()
        transitions[(source_state, character)].update(target_states)

    def _add_all_transitions(self, transitions, source_state, target_states):

        for character in self._get_alphabet():
            self._add_transition(transitions, source_state, character, target_states)

    def _add_transition_without_any(self, transitions, source_state, character, target_states):

        if character == self.ANY_INPUT:
            self._add_all_transitions(transitions, source_state, target_states)
        else:
            self._add_transition(transitions, source_state, character, target_states)

    def _create_levenshtein_dfa(self, word, distance, merge_split=False, transposition=False, repetitions=False):

        initial_state = (0,0)

        states = {initial_state}
        input_symbols = self._get_alphabet()
        final_states = set()
        transitions = dict()

        last_char = ''
        for position, character in enumerate(word):

            input_symbols.add(character)

            for error in range(distance + 1):

                current_state = (position, error)

                states.add(current_state)
                if (current_state, character) not in transitions:
                    transitions[(current_state, character)] = set()

                # Match
                self._add_transition(transitions, current_state, character, set([(position + 1, error)]))

                # Repetitions
                if repetitions:
                    ## Repetition state
                    self._add_transition(transitions, current_state, character, set([('rep', character, position + 1, error)]))
                    ## Repetition in the target word
                    self._add_transition(transitions, ('rep', character, position + 1, error), character, set([('rep', character, position + 1, error)]))
                    ## Repetition in the current word
                    if last_char and character == last_char:
                        self._add_transition(transitions, ('rep', character, position, error), self.EPSILON, set([('rep', character, position + 1, error)]))
                    ## Leave repetition state
                    self._add_transition(transitions, ('rep', character, position + 1, error), self.EPSILON, set([(position + 1, error)]))

                if error < distance:

                    # Insertion
                    self._add_transition(transitions, current_state, self.ANY_INPUT, set([(position, error + 1)]))

                    # Substitution
                    self._add_transition(transitions, current_state, self.ANY_INPUT, set([(position + 1, error + 1)]))

                    # Deletion
                    self._add_transition(transitions, current_state, self.EPSILON, set([(position + 1, error + 1)]))

                    # Merge and Split
                    if merge_split:
                        # Merge
                        self._add_transition(transitions, current_state, self.ANY_INPUT, set([(position + 2, error + 1)]))
                        # Split
                        self._add_transition(transitions, current_state, self.ANY_INPUT, set([(('split', position), error)]))
                        self._add_transition(transitions, (('split', position), error), self.ANY_INPUT, set([(position + 1, error + 1)]))

                    # Transposition
                    if transposition and last_char:
                        self._add_transition(transitions, (position - 1, error), character, set([(('transposition', position), error)]))
                        self._add_transition(transitions, (('transposition', position), error), last_char, set([(position + 1, error + 1)]))

            last_char = character

        for error in range(distance + 1):

            current_state = (len(word), error)
            states.add(current_state)

            if error < distance:
                self._add_transition(transitions, current_state, self.ANY_INPUT, set([(len(word), error + 1)]))

            final_states.add(current_state)

        ## remove epsilon and any from transitions
        transitions_without_epsilon_and_any = {}

        for (source_state, character), target_states in transitions.items():
            if character != self.EPSILON:
                self._add_transition_without_any(transitions_without_epsilon_and_any, source_state, character, target_states)
            else:
                ## get epsilon_closure of source_state
                epsilon_closure = target_states
                new_states = target_states
                while new_states:
                    curr_states = set()
                    for state in new_states:
                        curr_states.update(transitions.get((state, self.EPSILON), set()))

                    new_states = curr_states - epsilon_closure
                    epsilon_closure.update(curr_states)

                ## add transitions for epsilon closure
                for (reachable_state, character), new_target_states in transitions.items():
                    if character != self.EPSILON and reachable_state in epsilon_closure:
                        self._add_transition_without_any(transitions_without_epsilon_and_any, source_state, character, new_target_states)
                ## add to final states if final state in epsilon closure
                if epsilon_closure.intersection(final_states):
                    final_states.add(source_state)

        return nfa_determinization({
            'alphabet': input_symbols,
            'states': states,
            'initial_states': set([initial_state]),
            'accepting_states': final_states,
            'transitions': transitions_without_epsilon_and_any
        })
