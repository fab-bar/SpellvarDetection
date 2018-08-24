from spsim import SpSim


class UndirSpSim(SpSim):

    ### SpSim has hardcoded bow and eow characters
    __spsim_bow = '^'
    __spsim_eow = '$'

    def __init__(self,
                 examples=None,
                 ignore_case=True,
                 ignore_accents=True,
                 group_vowels=False,
                 no_empty=False,
                 bow=__spsim_bow,
                 eow=__spsim_eow):
        self.bow = bow
        self.eow = eow
        super().__init__(examples, ignore_case, ignore_accents, group_vowels, no_empty)

    def _prepare(self, a, b):

        ## we are cheating: '^' and '$' are hardcoded as bow and eow in SpSim
        ## so substitute with respective characters, if set

        if self.bow != UndirSpSim.__spsim_bow:
            a = a.replace(UndirSpSim.__spsim_bow, self.bow)
            b = b.replace(UndirSpSim.__spsim_bow, self.bow)
        if self.eow != UndirSpSim.__spsim_eow:
            a = a.replace(UndirSpSim.__spsim_eow, self.eow)
            b = b.replace(UndirSpSim.__spsim_eow, self.eow)

        return super()._prepare(a, b)

    def _get_diffs(self, a, b):

        for nchars, diff, ctxt in super()._get_diffs(a, b):
            yield nchars, frozenset(diff.split("\t")), ctxt
