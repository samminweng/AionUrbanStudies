# Helper function for LDA topic modeling
import math
import os
import re
import string
import sys
import pandas as pd
from nltk import word_tokenize, pos_tag, ngrams
import copy

from nltk.corpus import stopwords


class TopicKeywordClusterUtility:
    # Static variable
    stop_words = list(stopwords.words('english'))

    @staticmethod
    def compute_topic_coherence_score(doc_n_grams, topic_words):
        # Build a mapping of word and doc ids
        def _build_word_docIds(_doc_n_grams, _topic_words):
            _word_docIds = {}
            for _word in _topic_words:
                try:
                    _word_docIds.setdefault(_word, list())
                    # Get the number of docs containing the word
                    for _doc in _doc_n_grams:
                        _doc_id = _doc[0]
                        _n_grams = _doc[1]
                        _found = next((_n_gram for _n_gram in _n_grams if _word.lower() in _n_gram.lower()), None)
                        if _found:
                            _word_docIds[_word].append(_doc_id)
                except Exception as _err:
                    print("Error occurred! {err}".format(err=_err))
                    sys.exit(-1)
            return _word_docIds

        # # Get doc ids containing both word i and word j
        def _get_docIds_two_words(_docId_word_i, _docIds_word_j):
            return [_docId for _docId in _docId_word_i if _docId in _docIds_word_j]

        try:
            word_docs = _build_word_docIds(doc_n_grams, topic_words)
            score = 0
            for i in range(0, len(topic_words)):
                try:
                    word_i = topic_words[i]
                    docs_word_i = word_docs[word_i]
                    doc_count_word_i = len(docs_word_i)
                    assert doc_count_word_i > 0
                    for j in range(i + 1, len(topic_words)):
                        word_j = topic_words[j]
                        docs_word_j = word_docs[word_j]
                        doc_word_i_j = _get_docIds_two_words(docs_word_i, docs_word_j)
                        doc_count_word_i_j = len(doc_word_i_j)
                        assert doc_count_word_i_j >= 0
                        coherence_score = math.log((doc_count_word_i_j + 1) / (1.0 * doc_count_word_i))
                        score += coherence_score
                except Exception as _err:
                    print("Error occurred! {err}".format(err=_err))
                    sys.exit(-1)
            avg_score = score / (1.0 * len(topic_words))
            return avg_score, word_docs
        except Exception as _err:
            print("Error occurred! {err}".format(err=_err))

    # Generate n-gram candidates from a text (a list of sentences)
    @staticmethod
    def generate_n_gram_candidates(sentences, n_gram_range, is_check=True):
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
                            _word.lower() in TopicKeywordClusterUtility.stop_words or _pos_tag not in qualified_tags:
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

    @staticmethod
    def output_key_phrase_group_LDA_topics(clusters, cluster_no_list, folder, case_name):
        # Produce the output for each cluster
        results = list()
        for cluster_no in cluster_no_list:
            cluster = next(cluster for cluster in clusters if cluster['Cluster'] == cluster_no)
            result = {'cluster': cluster_no}
            # Added the grouped key phrase
            for i, group in enumerate(cluster['KeyPhrases']):
                # Convert the dictionary to a list
                word_docIds = group['word_docIds'].items()
                word_docIds = sorted(word_docIds, key=lambda w: w[1], reverse=True)
                result['group_' + str(i) + '_score'] = group['score']
                result['group_' + str(i)] = word_docIds

            # Added the LDA topics
            for i, topic in enumerate(cluster['LDATopics']):
                # Convert the dictionary to a list
                word_docIds = topic['word_docIds'].items()
                word_docIds = sorted(word_docIds, key=lambda w: w[1], reverse=True)
                result['LDATopic_' + str(i) + '_score'] = topic['score']
                result['LDATopic_' + str(i)] = word_docIds
            results.append(result)
        # Write to csv
        df = pd.DataFrame(results)
        path = os.path.join(folder, case_name + '_cluster_key_phrases_LDA_topics_summary.csv')
        df.to_csv(path, encoding='utf-8', index=False)

    # Get the topic words from each group of key phrases
    @staticmethod
    def collect_topic_words_from_key_phrases(key_phrases):
        # create a mapping between word and phrase
        def _create_word_phrase_list(_key_phrases):
            def _create_bi_grams(_words):
                if len(_words) >= 2:
                    _two_words = _words[-2:]
                    return [_two_words[0] + " " + _two_words[1]]
                return []

            _word_freq_dict = dict()
            # Collect word frequencies from the list of key phrases.
            for _key_phrase in _key_phrases:
                try:
                    _candidate_words = _key_phrase.split()
                    _candidate_words = list(map(lambda w: w.replace("'s", ""), _candidate_words))
                    # Include bi_grams
                    _candidate_words = _candidate_words + _create_bi_grams(_candidate_words)
                    # print(n_grams)
                    for _candidate in _candidate_words:
                        if _candidate not in _word_freq_dict:
                            _word_freq_dict[_candidate] = list()
                        # Get the frequencies
                        _word_freq_dict[_candidate].append(_key_phrase)
                except Exception as err:
                    print("Error occurred! {err}".format(err=err))
            _word_phrases = list()
            for _word, _phrases in _word_freq_dict.items():
                _word_phrases.append({'word': _word, 'phrases': _phrases})
            # Sort by freq and the number of docs
            _word_phrases = sorted(_word_phrases, key=lambda wf: (len(wf['phrases']), len(wf['word'].split(" "))),
                                   reverse=True)
            return _word_phrases

        # Update top word frequencies and pick up top words that increase the maximal coverage
        def pick_topic_words(_topic_words, _candidate_words, _top_n=5):
            # Go through top_words and avoid overlapping phrases
            # For example, 'traffic' and 'prediction' have overlap phrases 'traffic prediction'
            try:
                for i in range(0, _top_n):
                    cur_word = _topic_words[i]
                    cur_phrases = cur_word['phrases']
                    # Take out phrases of this word from other words
                    for j in range(i + 1, len(_topic_words)):
                        other_word = _topic_words[j]
                        # Remove duplicated word
                        other_word['phrases'] = list(
                            filter(lambda phrase: phrase not in cur_phrases, other_word['phrases']))
                    # Take out phrases of this    
                    for _candidate_word in _candidate_words:
                        # Remove candidate words containing phrases
                        _candidate_word['phrases'] = list(
                            filter(lambda phrase: phrase not in cur_phrases, _candidate_word['phrases']))
                # Add the candidate words if any top word is removed from the list
                _all_candidate_words = _topic_words + _candidate_words
                # Sort all the words by doc_ids and frequencies
                _all_candidate_words = sorted(_all_candidate_words,
                                              key=lambda wf: (len(wf['phrases']), len(wf['word'].split(" "))),
                                              reverse=True)
                return _all_candidate_words[:_top_n]
            except Exception as err:
                print("Error occurred! {err}".format(err=err))
                sys.exit(-1)

        word_phrases = _create_word_phrase_list(key_phrases)
        # Pick up top 5 frequent words
        top_n = 5
        word_phrases_clone = copy.deepcopy(word_phrases)
        topic_words = word_phrases_clone[:top_n]
        candidate_words = word_phrases_clone[top_n:]
        is_pick = True
        if is_pick:
            new_topic_words = []
            is_same = False
            iteration = 0
            while True:
                if iteration >= 5 or is_same:
                    topic_words = new_topic_words
                    break
                # Pass the copy array to the function to avoid change the values of 'top_word' 'candidate_words'
                new_topic_words = pick_topic_words(topic_words, candidate_words)
                # Check if new and old top words are the same
                is_same = True
                for new_word in new_topic_words:
                    found = next((w for w in topic_words if w['word'] == new_word['word']), None)
                    if not found:
                        is_same = is_same & False
                if not is_same:
                    def is_found(_word, _new_top_words):
                        _found = next((nw for nw in _new_top_words if nw['word'] == _word['word']), None)
                        if _found:
                            return True
                        return False

                    # Make a copy of wfl
                    word_phrases_clone = copy.deepcopy(word_phrases)
                    topic_words = list(filter(lambda word: is_found(word, new_topic_words), word_phrases_clone))
                    candidate_words = list(filter(lambda word: not is_found(word, new_topic_words), word_phrases_clone))
                    iteration += 1
            # assert len(top_words) >= 5, "topic word less than 5"
        # Return the top 3
        return list(map(lambda w: w['word'], topic_words[:5]))
