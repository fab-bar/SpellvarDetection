from spellvardetection.lib.simple_automata import dfa_intersection_language, nfa_determinization

class DictAutomaton:

    ANY_INPUT = '__ANY__'
    EPSILON = '__EPSILON__'

    def fuzzySearch(self, word, distance, merge_split=False, transposition=False, repetitions=False, strict_dist=False):

        lev_aut = self._create_levenshtein_dfa(word, distance, merge_split=merge_split, transposition=transposition, repetitions=repetitions)

        words = dfa_intersection_language(lev_aut, self.automaton, any_input=self.ANY_INPUT)

        if strict_dist:
            ## only keep words with the given distance
            words = filter(
                lambda word: distance == min([state[1] for state in word[0]]),
                words)

        return set(map(lambda word: word[1], words))

    def __init__(self, dictionary):

        ## create acceptor for dictionary

        initial_state = '_START_'

        final_states = set()
        transitions = dict()

        for word in dictionary:

            final_states.add(word)
            prefix = ''

            for character in word:

                if not prefix:
                    from_state = initial_state
                else:
                    from_state = prefix

                prefix = prefix + character

                if from_state not in transitions:
                    transitions[from_state] = {}
                transitions[from_state][character] = prefix


        self.automaton = {
            'initial_state': initial_state,
            'accepting_states': final_states,
            'transitions': transitions
        }

    def _add_transition(self, transitions, source_state, character, target_states):

        if source_state not in transitions:
            transitions[source_state] = {}
        if character not in transitions[source_state]:
            transitions[source_state][character] = set()
        transitions[source_state][character].update(target_states)

    def _create_levenshtein_dfa(self, word, distance, merge_split=False, transposition=False, repetitions=False):

        initial_state = (0,0, None)

        final_states = set()
        transitions = dict()

        last_char = ''
        for position, character in enumerate(word):

            for error in range(distance + 1):

                current_state = (position, error, None)

                # Match
                self._add_transition(transitions, current_state, character, set([(position + 1, error, None)]))

                # Repetitions
                if repetitions:
                    ## Repetition state
                    self._add_transition(transitions, current_state, character, set([(position + 1, error, ('rep', character))]))
                    ## Repetition in the target word
                    self._add_transition(transitions, (position + 1, error, ('rep', character)), character, set([(position + 1, error, ('rep', character))]))
                    ## Repetition in the current word
                    if last_char and character == last_char:
                        self._add_transition(transitions, (position, error, ('rep', character)), self.EPSILON, set([(position + 1, error, ('rep', character))]))
                    ## Leave repetition state
                    self._add_transition(transitions, (position + 1, error, ('rep', character)), self.EPSILON, set([(position + 1, error, None)]))

                if error < distance:

                    # Insertion
                    self._add_transition(transitions, current_state, self.ANY_INPUT, set([(position, error + 1, None)]))

                    # Substitution
                    self._add_transition(transitions, current_state, self.ANY_INPUT, set([(position + 1, error + 1, None)]))

                    # Deletion
                    self._add_transition(transitions, current_state, self.EPSILON, set([(position + 1, error + 1, None)]))

                    # Merge and Split
                    if merge_split:
                        # Merge
                        self._add_transition(transitions, current_state, self.ANY_INPUT, set([(position + 2, error + 1, None)]))
                        # Split
                        self._add_transition(transitions, current_state, self.ANY_INPUT, set([(position, error, 'split')]))
                        self._add_transition(transitions, (position, error, 'split'), self.ANY_INPUT, set([(position + 1, error + 1, None)]))

                    # Transposition
                    if transposition and last_char:
                        self._add_transition(transitions, (position - 1, error, None), character, set([(position, error, 'transposition')]))
                        self._add_transition(transitions, (position, error, 'transposition'), last_char, set([(position + 1, error + 1, None)]))

            last_char = character

        for error in range(distance + 1):

            current_state = (len(word), error, None)

            if error < distance:
                self._add_transition(transitions, current_state, self.ANY_INPUT, set([(len(word), error + 1, None)]))

            final_states.add(current_state)

        return nfa_determinization({
            'initial_states': set([initial_state]),
            'accepting_states': final_states,
            'transitions': transitions
        }, any_input=self.ANY_INPUT, epsilon=self.EPSILON)
