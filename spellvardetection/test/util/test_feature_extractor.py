import unittest
import math

from spellvardetection.util.feature_extractor import SurfaceExtractor, ContextExtractor, NGramExtractor

class TestSurfaceExtractor(unittest.TestCase):

    def setUp(self):

        self.data_point = ('test', 'fest')

    ## test cache of the mixin
    def test_feature_extractor_cache(self):

        ext = SurfaceExtractor()

        ## test cache hit
        ext.setFeatureCache({
            ext._getDataKey(self.data_point): 'a'
        })

        self.assertEquals(
            ext.extractFeaturesFromDatapoint(self.data_point),
            'a'
        )

        ## test empty cache
        ext.setFeatureCache()
        self.assertEquals(
            set(ext.extractFeaturesFromDatapoint(self.data_point)),
            set([('ft',), ('$$', 'ft'), ('ft', 'ee'), ('$$', 'ft', 'ee'), ('ft', 'ee', 'ss')])
        )
        self.assertEquals(
            {key: set(value) for key,value in ext.feature_cache.items()},
            {ext._getDataKey(self.data_point):
            set([('ft',), ('$$', 'ft'), ('ft', 'ee'), ('$$', 'ft', 'ee'), ('ft', 'ee', 'ss')])}
        )

    def test_feature_extractor_cache_with_key(self):

        ext = SurfaceExtractor()

        ## test cache hit
        ext.setFeatureCache({
            ext._getDataKey(self.data_point): {'ngrams': 'a'}
        }, 'ngrams')

        self.assertEquals(
            ext.extractFeaturesFromDatapoint(self.data_point),
            'a'
        )

        ## test empty cache
        ext.setFeatureCache(key='ngrams')
        self.assertEquals(
            set(ext.extractFeaturesFromDatapoint(self.data_point)),
            set([('ft',), ('$$', 'ft'), ('ft', 'ee'), ('$$', 'ft', 'ee'), ('ft', 'ee', 'ss')])
        )

        ## now with cache hit
        self.assertEquals(
            set(ext.extractFeaturesFromDatapoint(self.data_point)),
            set([('ft',), ('$$', 'ft'), ('ft', 'ee'), ('$$', 'ft', 'ee'), ('ft', 'ee', 'ss')])
        )

    def test_feature_extractor_pickle(self):

        ext = SurfaceExtractor()

        ext.setFeatureCache({
            ext._getDataKey(self.data_point): {'ngrams': 'a'}
        }, 'ngrams')

        self.assertTrue('feature_cache' not in ext.__getstate__())

    def test_extract_all_ngrams(self):

        ext = SurfaceExtractor(only_mismatch_ngrams=False)
        self.assertEquals(
            set(ext.extractFeaturesFromDatapoint(self.data_point)),
            set([
                ('$$',), ('ft',), ('ee',), ('ss',), ('tt',),
                ('$$', 'ft'), ('ft', 'ee'), ('ee', 'ss'), ('ss', 'tt'), ('tt', '$$'),
                ('$$', 'ft', 'ee'), ('ft', 'ee', 'ss'), ('ee', 'ss', 'tt'), ('ss', 'tt', '$$')
            ])
        )

    def test_extract_mismatch_ngrams(self):

        ext = SurfaceExtractor(only_mismatch_ngrams=True)
        self.assertEquals(
            set(ext.extractFeaturesFromDatapoint(self.data_point)),
            set([('ft',), ('$$', 'ft'), ('ft', 'ee'), ('$$', 'ft', 'ee'), ('ft', 'ee', 'ss')])
        )


    def test_extract_all_unigrams(self):

        ext = SurfaceExtractor(max_ngram_size=1)
        self.assertEquals(
            set(ext.extractFeaturesFromDatapoint(self.data_point)),
            set([('ft',)])
        )


    def test_extract_all_bigrams(self):

        ext = SurfaceExtractor(min_ngram_size=2, max_ngram_size=2)
        self.assertEquals(
            set(ext.extractFeaturesFromDatapoint(self.data_point)),
            set([('$$', 'ft'), ('ft', 'ee')])
        )


class TestContextExtractor(unittest.TestCase):

    def setUp(self):

        self.data_point = ('test', 'fest')

    def test_extract_orthogonal_vectors(self):

        ext = ContextExtractor({'test': [0,2], 'fest': [2,0]})

        self.assertEquals(
            ext.extractFeaturesFromDatapoint(self.data_point),
            0
        )


    def test_extract_equal_vectors(self):

        ext = ContextExtractor({'test': [2,0], 'fest': [2,0]})

        self.assertEquals(
            ext.extractFeaturesFromDatapoint(self.data_point),
            1
        )


    def test_extract_with_missing_vector(self):

        ext = ContextExtractor({'test': [2,0], 'fest': None})

        self.assertTrue(math.isnan(ext.extractFeaturesFromDatapoint(self.data_point)))


class TestNGramExtractor(unittest.TestCase):

    def setUp(self):

        self.data_point = ('insurgents', 'killed', 'in', 'ongoing', 'fighting')

    def test_padding_sequence(self):

        ext = NGramExtractor(min_ngram_size=2, max_ngram_size=2, skip_size=0, bow='<BOS>', eow='<EOS>', pad_ngrams=False)

        self.assertEquals(
            ext.extractFeaturesFromDatapoint(self.data_point),
            set([
                ('<BOS>', 'insurgents'),
                ('insurgents', 'killed'),
                ('killed', 'in'),
                ('in', 'ongoing'),
                ('ongoing', 'fighting'),
                ('fighting', '<EOS>')
            ])
        )

    def test_padding_ngrams(self):

        ext = NGramExtractor(min_ngram_size=2, max_ngram_size=2, skip_size=0, bow='<BOS>', eow='<EOS>', pad_ngrams=True)

        self.assertEquals(
            ext.extractFeaturesFromDatapoint(self.data_point),
            set([
                ('<BOS>insurgents', 'killed'),
                ('killed', 'in'),
                ('in', 'ongoing'),
                ('ongoing', 'fighting<EOS>'),
            ])
        )

    def test_extract_bi_grams(self):

        ext = NGramExtractor(min_ngram_size=2, max_ngram_size=2, skip_size=0, bow='', eow='')

        self.assertEquals(
            ext.extractFeaturesFromDatapoint(self.data_point),
            set([
                ('insurgents', 'killed'),
                ('killed', 'in'),
                ('in', 'ongoing'),
                ('ongoing', 'fighting'),
            ])
        )

    def test_extract_2_skip_bi_grams(self):

        ext = NGramExtractor(min_ngram_size=2, max_ngram_size=2, skip_size=2, gap='', bow='', eow='')

        self.assertEquals(
            ext.extractFeaturesFromDatapoint(self.data_point),
            set([
                ('insurgents', 'killed'),
                ('insurgents', 'in'),
                ('insurgents', 'ongoing'),
                ('killed', 'in'),
                ('killed', 'ongoing'),
                ('killed', 'fighting'),
                ('in', 'ongoing'),
                ('in', 'fighting'),
                ('ongoing', 'fighting'),
            ])
        )

    def test_extract_tri_grams(self):

        ext = NGramExtractor(min_ngram_size=3, max_ngram_size=3, skip_size=0, bow='', eow='')

        self.assertEquals(
            ext.extractFeaturesFromDatapoint(self.data_point),
            set([
                ('insurgents', 'killed', 'in'),
                ('killed', 'in', 'ongoing'),
                ('in', 'ongoing', 'fighting')
            ])
        )

    def test_extract_2_skip_tri_grams(self):

        ext = NGramExtractor(min_ngram_size=3, max_ngram_size=3, skip_size=2, gap='', bow='', eow='')

        self.assertEquals(
            ext.extractFeaturesFromDatapoint(self.data_point),
            set([
                ('insurgents', 'killed', 'in'),
                ('insurgents', 'killed', 'ongoing'),
                ('insurgents', 'killed', 'fighting'),
                ('insurgents', 'in', 'ongoing'),
                ('insurgents', 'in', 'fighting'),
                ('insurgents', 'ongoing', 'fighting'),
                ('killed', 'in', 'ongoing'),
                ('killed', 'in', 'fighting'),
                ('killed', 'ongoing', 'fighting'),
                ('in', 'ongoing', 'fighting')
            ])
        )

    def test_extract_1_skip_bi_grams_with_gap_and_ngrampadding_love(self):

        ext = NGramExtractor(min_ngram_size=2, max_ngram_size=2, skip_size=1, gap='|', bow='$', eow='$', pad_ngrams=True)

        self.assertEquals(
            ext.extractFeaturesFromDatapoint('love'),
            set([
                ('$l', 'o'),
                ('o', 'v'),
                ('v', 'e$'),
                ('l', '|', 'v'),
                ('o', '|', 'e')
            ])
        )

    def test_extract_1_skip_bi_grams_with_gap_and_ngrampadding_looove(self):

        ext = NGramExtractor(min_ngram_size=2, max_ngram_size=2, skip_size=1, gap='|', bow='$', eow='$', pad_ngrams=True)

        self.assertEquals(
            ext.extractFeaturesFromDatapoint('looove'),
            set([
                ('$l', 'o'),
                ('o', 'o'),
                ('o', 'v'),
                ('v', 'e$'),
                ('l', '|', 'o'),
                ('o', '|', 'o'),
                ('o', '|', 'v'),
                ('o', '|', 'e')
            ])
        )

    def test_extract_1_skip_bi_grams_with_gap_and_ngrampadding_car(self):

        ext = NGramExtractor(min_ngram_size=2, max_ngram_size=2, skip_size=1, gap='|', bow='$', eow='$', pad_ngrams=True)

        self.assertEquals(
            ext.extractFeaturesFromDatapoint('car'),
            set([
                ('$c', 'a'),
                ('a', 'r$'),
                ('c', '|', 'r')
            ])
        )

    def test_extract_1_skip_bi_grams_with_gap_and_ngrampadding_cat(self):

        ext = NGramExtractor(min_ngram_size=2, max_ngram_size=2, skip_size=1, gap='|', bow='$', eow='$', pad_ngrams=True)

        self.assertEquals(
            ext.extractFeaturesFromDatapoint('cat'),
            set([
                ('$c', 'a'),
                ('a', 't$'),
                ('c', '|', 't'),
            ])
        )

    def test_extract_all_ngrams(self):

        ext = NGramExtractor(min_ngram_size=3, max_ngram_size=float('inf'), skip_size=0, gap='', bow='', eow='')

        self.assertEquals(
            ext.extractFeaturesFromDatapoint('#comparable#'),
            set([
                tuple('#comparable#'),
                tuple('#comparable'), tuple('comparable#'),
                tuple('#comparabl'), tuple('comparable'), tuple('omparable#'),
                tuple('#comparab'), tuple('comparabl'), tuple('omparable'), tuple('mparable#'),
                tuple('#compara'), tuple('comparab'), tuple('omparabl'), tuple('mparable'), tuple('parable#'),
                tuple('#compar'), tuple('compara'), tuple('omparab'), tuple('mparabl'), tuple('parable'), tuple('arable#'),
                tuple('#compa'), tuple('compar'), tuple('ompara'), tuple('mparab'), tuple('parabl'), tuple('arable'), tuple('rable#'),
                tuple('#comp'), tuple('compa'), tuple('ompar'), tuple('mpara'), tuple('parab'), tuple('arabl'), tuple('rable'), tuple('able#'),
                tuple('#com'), tuple('comp'), tuple('ompa'), tuple('mpar'), tuple('para'), tuple('arab'), tuple('rabl'), tuple('able'), tuple('ble#'),
                tuple('#co'), tuple('com'), tuple('omp'), tuple('mpa'), tuple('par'), tuple('ara'), tuple('rab'), tuple('abl'), tuple('ble'), tuple('le#')
            ])
        )

    def test_extract_all_ngrams_with_padding(self):

        ext = NGramExtractor(min_ngram_size=3, max_ngram_size=float('inf'), skip_size=0, gap='', bow='#', eow='#')

        self.assertEquals(
            ext.extractFeaturesFromDatapoint('comparable'),
            set([
                tuple('#comparable#'),
                tuple('#comparable'), tuple('comparable#'),
                tuple('#comparabl'), tuple('comparable'), tuple('omparable#'),
                tuple('#comparab'), tuple('comparabl'), tuple('omparable'), tuple('mparable#'),
                tuple('#compara'), tuple('comparab'), tuple('omparabl'), tuple('mparable'), tuple('parable#'),
                tuple('#compar'), tuple('compara'), tuple('omparab'), tuple('mparabl'), tuple('parable'), tuple('arable#'),
                tuple('#compa'), tuple('compar'), tuple('ompara'), tuple('mparab'), tuple('parabl'), tuple('arable'), tuple('rable#'),
                tuple('#comp'), tuple('compa'), tuple('ompar'), tuple('mpara'), tuple('parab'), tuple('arabl'), tuple('rable'), tuple('able#'),
                tuple('#com'), tuple('comp'), tuple('ompa'), tuple('mpar'), tuple('para'), tuple('arab'), tuple('rabl'), tuple('able'), tuple('ble#'),
                tuple('#co'), tuple('com'), tuple('omp'), tuple('mpa'), tuple('par'), tuple('ara'), tuple('rab'), tuple('abl'), tuple('ble'), tuple('le#')
            ])
        )

    def test_extract_all_ngrams(self):

        ext = NGramExtractor(min_ngram_size=1, max_ngram_size=2, skip_size=1, gap='', bow='', eow='')

        self.assertEquals(
            ext.extractFeaturesFromDatapoint('#abcd#'),
            set([
                tuple('#a'), tuple('#b'), tuple('ab'), tuple('ac'), tuple('bc'), tuple('bd'), tuple('cd'), tuple('c#'), tuple('d#'),
                tuple('#'), tuple('a'), tuple('b'), tuple('c'), tuple('d'), tuple('#'),
            ])
        )
