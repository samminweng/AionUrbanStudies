# Helper function for LDA topic modeling
import math
import re
import string
import sys

from nltk import sent_tokenize, word_tokenize, pos_tag, ngrams

from BERTModelDocClusterUtility import BERTModelDocClusterUtility


class ClusterTopicUtility:
    @staticmethod
    def compute_topic_coherence_score(doc_n_gram_list, topic_words):
        # Get the number of docs containing the word
        def _count_docs_by_word(_doc_n_gram_list, _word):
            _count = 0
            for _doc_n_grams in _doc_n_gram_list:
                _found = next((_n_gram for _n_gram in _doc_n_grams if _n_gram.lower() == _word.lower()), None)
                if _found:
                    _count += 1
            return _count

        # Get the number of docs containing both word i and word j
        def _count_docs_by_two_words(_doc_n_gram_list, _word_i, _word_j):
            _count = 0
            for _doc_n_grams in _doc_n_gram_list:
                _found_i = next((_n_gram for _n_gram in _doc_n_grams if _n_gram.lower() == _word_i.lower()), None)
                _found_j = next((_n_gram for _n_gram in _doc_n_grams if _n_gram.lower() == _word_j.lower()), None)
                if _found_i and _found_j:
                    _count += 1
            return _count

        score = 0
        for i in range(0, len(topic_words)):
            try:
                word_i = topic_words[i]
                doc_count_word_i = _count_docs_by_word(doc_n_gram_list, word_i)
                assert doc_count_word_i > 0
                for j in range(i+1, len(topic_words)):
                    word_j = topic_words[j]
                    doc_count_word_i_j = _count_docs_by_two_words(doc_n_gram_list, word_i, word_j)
                    assert doc_count_word_i_j >= 0
                    coherence_score = math.log((doc_count_word_i_j + 1)/(1.0 * doc_count_word_i))
                    score += coherence_score
            except Exception as _err:
                print("Error occurred! {err}".format(err=_err))
                sys.exit(-1)
        avg_score = score / (1.0*len(topic_words))
        return avg_score

    # Generate n-gram candidates from a text (a list of sentences)
    @staticmethod
    def generate_n_gram_candidates(sentences, n_gram_range):
        # Check if n_gram candidate does not have stop words, punctuation or non-words
        def _is_qualified(_n_gram):  # _n_gram is a list of tuple (word, tuple)
            try:
                qualified_tags = ['NN', 'NNS', 'JJ', 'NNP']
                # # # Check if there is any noun
                nouns = list(filter(lambda _n: _n[1].startswith('NN'), _n_gram))
                if len(nouns) == 0:
                    return False
                # # Check the last word is a nn or nns
                if _n_gram[-1][1] not in ['NN', 'NNS']:
                    return False
                # Check if all words are not stop word or punctuation or non-words
                for _i, _n in enumerate(_n_gram):
                    _word = _n[0]
                    _pos_tag = _n[1]
                    if bool(re.search(r'\d|[^\w]', _word.lower())) or _word.lower() in string.punctuation or \
                            _word.lower() in BERTModelDocClusterUtility.stop_words or _pos_tag not in qualified_tags:
                        return False
                # n-gram is qualified
                return True
            except Exception as _err:
                print("Error occurred! {err}".format(err=_err))

        # Convert n_gram tuples (pos tag and words) to a list of singular words
        def _convert_n_gram_to_words(_n_gram):
            _lemma_words = list()
            for _gram in _n_gram:
                _word = _gram[0]
                _pos_tag = _gram[1]
                _lemma_words.append(_word)
            return " ".join(_lemma_words)

        candidates = list()
        # Extract n_gram from each sentence
        for i, sentence in enumerate(sentences):
            try:
                words = word_tokenize(sentence)
                pos_tags = pos_tag(words)
                # Pass pos tag tuple (word, pos-tag) of each word in the sentence to produce n-grams
                _n_grams = list(ngrams(pos_tags, n_gram_range))
                # Filter out not qualified n_grams that contain stopwords or the word is not alpha_numeric
                for _n_gram in _n_grams:
                    if _is_qualified(_n_gram):
                        n_gram_words = _convert_n_gram_to_words(_n_gram)
                        candidates.append(n_gram_words)  # Convert n_gram (a list of words) to a string
            except Exception as _err:
                print("Error occurred! {err}".format(err=_err))
        return candidates