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
    def collect_topic_words_from_key_phrases(key_phrases, doc_n_grams):
        # create a mapping between word and frequencies
        def create_word_freq_list(_key_phrases, _doc_n_grams):
            def _create_bi_grams(_words):
                _bi_grams = list()
                if len(_words) == 2:
                    _bi_grams.append(_words[0] + " " + _words[1])
                elif len(_words) == 3:
                    # _bi_grams.append(_words[0] + " " + _words[1])
                    _bi_grams.append(_words[1] + " " + _words[2])
                return _bi_grams

            # Get the docs containing the word
            def _get_doc_ids_by_key_phrase(_key_phrase, _doc_n_grams):
                doc_ids = list()
                for doc in _doc_n_grams:
                    doc_id = doc[0]
                    n_grams = doc[1]
                    found = list(filter(lambda n_gram: _key_phrase.lower() in n_gram.lower(), n_grams))
                    if len(found) > 0:
                        doc_ids.append(doc_id)
                return doc_ids

            _word_freq_list = list()
            # Collect word frequencies from the list of key phrases.
            for key_phrase in key_phrases:
                try:
                    key_phrase_doc_ids = _get_doc_ids_by_key_phrase(key_phrase, _doc_n_grams)
                    words = key_phrase.split()
                    n_grams = words + _create_bi_grams(words)
                    # print(n_grams)
                    for n_gram in n_grams:
                        r = len(n_gram.split(" "))
                        found = next((wf for wf in _word_freq_list if wf['word'].lower() == n_gram.lower()), None)
                        if not found:
                            wf = {'word': n_gram.lower(), 'freq': 1, 'range': r, 'doc_ids': key_phrase_doc_ids}
                            if n_gram.isupper():
                                wf['word'] = n_gram
                            _word_freq_list.append(wf)
                        else:
                            # Updated doc id
                            found['doc_ids'] = found['doc_ids'] + key_phrase_doc_ids
                            # Remove duplicates
                            found['doc_ids'] = list(dict.fromkeys(found['doc_ids']))
                            found['freq'] = len(found['doc_ids'])
                except Exception as err:
                    print("Error occurred! {err}".format(err=err))
            return _word_freq_list

        # Update top word frequencies and pick up top words that increase the maximal coverage
        def pick_top_words(_top_words, _candidate_words, _top_n):
            # Go through top_words and check if other top words can be merged.
            # For example, 'traffic prediction' can be merged to 'traffic'
            try:
                for i in range(0, len(_top_words)):
                    top_word = _top_words[i]
                    words = top_word['word'].split(" ")
                    for j in range(i + 1, len(_top_words)):
                        other_word = _top_words[j]
                        # Remove duplicated word
                        _found = list(filter(lambda w: w in words, other_word['word'].split(" ")))
                        if len(_found) > 0:
                            other_word['doc_ids'] = list()
                # Remove no associated doc ids
                _top_words = list(filter(lambda w: len(w['doc_ids']) > 0, top_words))
                for top_word in _top_words:
                    words = top_word['word'].split(" ")
                    for word in words:
                        # Remove candidate words containing words
                        _candidate_words = list(filter(lambda cw: word not in cw['word'], _candidate_words))
                    # # # # Go through each candidate words and pick up
                    # for candidate_word in _candidate_words:
                    #     # Update the doc_id from
                    #     candidate_word['doc_ids'] = list(filter(lambda _id: _id not in top_word['doc_ids'], candidate_word['doc_ids']))
                # Add the candidate words if any top word is removed from the list
                all_words = _top_words + _candidate_words
                # Sort all the words by doc_ids and frequencies
                all_words = sorted(all_words, key=lambda wf: (len(wf['doc_ids']), wf['range'], wf['freq']),
                                   reverse=True)
                return all_words[:_top_n]
            except Exception as err:
                print("Error occurred! {err}".format(err=err))

        # Check if
        def is_found(_word, _new_top_words):
            _found = next((nw for nw in _new_top_words if nw['word'] == _word['word']), None)
            if _found:
                return True
            return False

        word_freq_list = create_word_freq_list(key_phrases, doc_n_grams)
        # Pick up top 5 frequent words
        top_n = 5
        # Sort by freq and the number of docs
        word_freq_list = sorted(word_freq_list, key=lambda wf: (wf['freq'], wf['range'], len(wf['doc_ids'])),
                                reverse=True)
        print(word_freq_list)
        word_freq_clone = copy.deepcopy(word_freq_list)
        top_words = word_freq_clone[:top_n]
        candidate_words = word_freq_clone[top_n:]
        is_pick = True
        if is_pick:
            new_top_words = copy.deepcopy(top_words)
            is_same = False
            iteration = 0
            while True:
                if iteration >= 10 or is_same:
                    top_words = new_top_words
                    break
                # Pass the copy array to the function to avoid change the values of 'top_word' 'candidate_words'
                new_top_words = pick_top_words(top_words, candidate_words, top_n)
                # Check if new and old top words are the same
                is_same = True
                for new_word in new_top_words:
                    found = next((w for w in top_words if w['word'] == new_word['word']), None)
                    if not found:
                        is_same = is_same & False
                if not is_same:
                    # Make a copy of wfl
                    word_freq_clone = copy.deepcopy(word_freq_list)
                    # Replace the old top words with new top words
                    top_words = list(filter(lambda word: is_found(word, new_top_words), word_freq_clone))
                    candidate_words = list(filter(lambda word: not is_found(word, new_top_words), word_freq_clone))
                    iteration += 1
            # assert len(top_words) >= 5, "topic word less than 5"
        # Sort topic words by frequencies
        top_words = sorted(top_words, key=lambda w: w['freq'], reverse=True)
        # Return the top 3
        return list(map(lambda w: w['word'], top_words[:5]))
