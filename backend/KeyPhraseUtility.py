import copy
import getpass
import math
import os
import re
import string
import sys
from functools import reduce
from pathlib import Path

import hdbscan
import nltk
import umap
from nltk import word_tokenize, sent_tokenize, ngrams, pos_tag
import pandas as pd
import numpy as np
# Load function words
from nltk.corpus import stopwords
from sklearn.metrics.pairwise import cosine_similarity, pairwise_distances
from nltk.stem import WordNetLemmatizer

# Set NLTK data path
from BERTModelDocClusterUtility import BERTModelDocClusterUtility

nltk_path = os.path.join('/Scratch', getpass.getuser(), 'nltk_data')
if os.name == 'nt':
    nltk_path = os.path.join("C:", os.sep, "Users", getpass.getuser(), "nltk_data")
nltk.download('punkt', download_dir=nltk_path)
nltk.download('wordnet', download_dir=nltk_path)
nltk.download('stopwords', download_dir=nltk_path)
nltk.download('averaged_perceptron_tagger', download_dir=nltk_path)  # POS tags
# Append NTLK data path
nltk.data.path.append(nltk_path)


# Helper function for cluster Similarity
class KeyPhraseUtility:
    stop_words = list(stopwords.words('english'))

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

    @staticmethod
    # Generate Collocation using regular expression patterns
    def generate_collocation_candidates(sentences):
        candidates = list()
        # Extract n_gram from each sentence
        for i, sentence in enumerate(sentences):
            try:
                pos_tags = pos_tag(sentence)
                # Pass pos tag tuple (word, pos-tag) of each word in the sentence to produce n-grams
                sentence_text = " ".join(list(map(lambda n: n[0] + '_' + n[1], pos_tags)))
                # print(sentence_text)
                # Use the regular expression to obtain n_gram
                # Patterns: (1) J + N (2) N + N (3) J + J + N + N (4) J + and + J + N + N
                sentence_words = list()
                pattern = r'((\s*\w+(\-\w+)*_NN[P]*[S]*\s*(\'s_POS)*\s*){2,3})' \
                          r'|((\w+(\-\w+)*_JJ\s+){1,2}(\w+(\-\w+)*_NN[P]*[S]*\s*(\'s_POS)*\s*){1,2})' \
                          r'|((\w+(\-\w+)*_JJ\s+)(and_CC\s+)(\w+(\-\w+)*_JJ\s+)(\w+(\-\w+)*_NN[P]*[S]*\s*(\'s_POS)*\s*){1,2})'
                matches = re.finditer(pattern, sentence_text)
                for match_obj in matches:
                    try:
                        n_gram = match_obj.group(0)
                        n_gram = n_gram.replace(" 's_POS", "'s")
                        n_gram = n_gram.replace("_CC", "")
                        n_gram = n_gram.replace("_JJ", "")
                        n_gram = n_gram.replace("_NNPS", "")
                        n_gram = n_gram.replace("_NNP", "")
                        n_gram = n_gram.replace("_NNS", "")
                        n_gram = n_gram.replace("_NN", "")
                        n_gram = n_gram.strip()
                        sentence_words.append(n_gram)
                    except Exception as _err:
                        print("Error occurred! {err}".format(err=_err))
                for word in sentence_words:
                    found = next((cw for cw in candidates if cw.lower() == word.lower()), None)
                    if not found:
                        candidates.append(word)
            except Exception as err:
                print("Error occurred! {err}".format(err=err))
        return candidates

    # Get single_word by using standard TF-IDF for each doc in
    @staticmethod
    def generate_tfidf_terms(cluster_docs, folder):
        # Generate n-gram of a text and avoid stop
        def _generate_n_gram_candidates(_sentences, _n_gram_range):
            def _is_qualified(_n_gram):  # _n_gram is a list of tuple (word, tuple)
                try:
                    qualified_tags = ['NN', 'NNS', 'NNP', 'NNPS']
                    # Check if all words are not stop word or punctuation or non-words
                    for _i, _n in enumerate(_n_gram):
                        _word = _n[0]
                        _pos_tag = _n[1]
                        if bool(re.search(r'\d|[^\w]', _word.lower())) or _word.lower() in string.punctuation or \
                                _word.lower() in KeyPhraseUtility.stop_words:  # or _pos_tag not in qualified_tags:
                            return False
                    # n-gram is qualified
                    return True
                except Exception as err:
                    print("Error occurred! {err}".format(err=err))

            candidates = list()
            # Extract n_gram from each sentence
            for i, sentence in enumerate(_sentences):
                try:
                    pos_tags = pos_tag(sentence)
                    # Pass pos tag tuple (word, pos-tag) of each word in the sentence to produce n-grams
                    n_grams = list(ngrams(pos_tags, _n_gram_range))
                    sentence_candidates = list()
                    # Filter out not qualified n_grams that contain stopwords or the word is not alpha_numeric
                    for n_gram in n_grams:
                        if _is_qualified(n_gram):
                            n_gram_text = " ".join(list(map(lambda n: n[0], n_gram)))
                            sentence_candidates.append(n_gram_text)  # Convert n_gram (a list of words) to a string
                    candidates = candidates + sentence_candidates
                except Exception as _err:
                    print("Error occurred! {err}".format(err=_err))
            return candidates

        # Create frequency matrix to track the frequencies of a n-gram in
        def _create_frequency_matrix(_docs, _n_gram_range):
            # Vectorized the clustered doc text and Keep the Word case unchanged
            frequency_matrix = []
            for doc in _docs:
                _doc_id = doc['DocId']  # doc id
                doc_text = BERTModelDocClusterUtility.preprocess_text(doc['Abstract'])
                sentences = list()
                for sentence in sent_tokenize(doc_text):
                    tokens = word_tokenize(sentence)
                    sentences.append(tokens)
                freq_table = {}
                candidates = _generate_n_gram_candidates(sentences, _n_gram_range)
                for candidate in candidates:
                    term = candidate.lower()
                    if candidate.isupper():
                        term = candidate
                    if term in freq_table:
                        freq_table[term] += 1
                    else:
                        freq_table[term] = 1
                frequency_matrix.append({'doc_id': _doc_id, 'freq_table': freq_table})
            return frequency_matrix

        # Compute TF score
        def _compute_tf_matrix(_freq_matrix):
            _tf_matrix = {}
            # Compute tf score for each cluster (doc) in the corpus
            for _row in _freq_matrix:
                _doc_id = _row['doc_id']  # Doc id is the cluster no
                _freq_table = _row['freq_table']  # Store the frequencies of each word in the doc
                _tf_table = {}  # TF score of each word (such as 1, 2, 3-gram) in the doc
                _total_terms_in_doc = reduce(lambda total, f: total + f, _freq_table.values(), 0)
                # Adjusted for total number of words in doc
                for _term, _freq in _freq_table.items():
                    # frequency of a word in doc / total number of words in doc
                    _tf_table[_term] = _freq / _total_terms_in_doc
                _tf_matrix[_doc_id] = _tf_table
            return _tf_matrix

        # Collect the table to store the mapping between word to a list of clusters
        def _create_occ_per_term(_freq_matrix):
            _occ_table = {}  # Store the mapping between a word and its doc ids
            for _row in _freq_matrix:
                _doc_id = _row['doc_id']  # Doc id is the cluster no
                _freq_table = _row['freq_table']  # Store the frequencies of each word in the doc
                for _term, _count in _freq_table.items():
                    if _term in _occ_table:  # Add the table if the word appears in the doc
                        _occ_table[_term].add(_doc_id)
                    else:
                        _occ_table[_term] = {_doc_id}
            return _occ_table

        # Compute IDF scores
        def _compute_idf_matrix(_freq_matrix, _occ_per_term):
            _total_cluster = len(_freq_matrix)  # Total number of clusters in the corpus
            _idf_matrix = {}  # Store idf scores for each doc
            for _row in _freq_matrix:
                _doc_id = _row['doc_id']  # Doc id is the cluster no
                _freq_table = _row['freq_table']  # Store the frequencies of each word in the doc
                _idf_table = {}
                for _term in _freq_table.keys():
                    _counts = len(_occ_per_term[_term])  # Number of clusters the word appears
                    _idf_table[_term] = math.log10(_total_cluster / float(_counts))
                _idf_matrix[_doc_id] = _idf_table  # Idf table stores each word's idf scores
            return _idf_matrix

        # Compute tf-idf score matrix
        def _compute_tf_idf_matrix(_tf_matrix, _idf_matrix, _freq_matrix, _occ_per_term):
            _tf_idf_matrix = list()
            # Compute tf-idf score for each cluster
            for _doc_id, _tf_table in _tf_matrix.items():
                # Compute tf-idf score of each word in the cluster
                _idf_table = _idf_matrix[_doc_id]  # idf table stores idf scores of the doc (doc_id)
                # Get freq table of the cluster
                _freq_table = next(f for f in _freq_matrix if f['doc_id'] == _doc_id)['freq_table']
                _tf_idf_list = []
                for _term, _tf_score in _tf_table.items():  # key is word, value is tf score
                    try:
                        _idf_score = _idf_table[_term]  # Get idf score of the word
                        _freq = _freq_table[_term]  # Get the frequencies of the word in doc_id
                        _doc_ids = sorted(list(_occ_per_term[_term]))  # Get the clusters that the word appears
                        _score = float(_tf_score * _idf_score)
                        _tf_idf_list.append({'term': _term, 'score': _score, 'freq': _freq, 'doc_ids': _doc_ids})
                    except Exception as _err:
                        print("Error occurred! {err}".format(err=_err))
                # Sort tf_idf_list by tf-idf score
                _term_list = sorted(_tf_idf_list, key=lambda t: t['score'], reverse=True)
                _tf_idf_matrix.append({'doc_id': _doc_id, 'terms': _term_list})
                # Write the selected output to csv files
                if _doc_id in [206, 325, 523]:
                    # Write to a list
                    _term_df = pd.DataFrame(_term_list, columns=['term', 'score', 'freq', 'doc_ids'])
                    # Write the topics results to csv
                    _term_df.to_csv(os.path.join(folder, 'TF-IDF_doc_terms_' + str(_doc_id) + '.csv'), encoding='utf-8',
                                    index=False)
            return _tf_idf_matrix

        try:
            # 2. Create the Frequency matrix of the words in each document (a cluster of articles)
            freq_matrix = _create_frequency_matrix(cluster_docs, 1)
            # # 3. Compute Term Frequency (TF) and generate a matrix
            # # Term frequency (TF) is the frequency of a word in a document divided by total number of words in the document.
            tf_matrix = _compute_tf_matrix(freq_matrix)
            # # 4. Create the table to map the word to a list of documents
            occ_per_term = _create_occ_per_term(freq_matrix)
            # # 5. Compute IDF (how common or rare a word is) and output the results as a matrix
            idf_matrix = _compute_idf_matrix(freq_matrix, occ_per_term)
            # # Compute tf-idf matrix
            terms_list = _compute_tf_idf_matrix(tf_matrix, idf_matrix, freq_matrix, occ_per_term)
            return terms_list  # Return a list of dicts
        except Exception as err:
            print("Error occurred! {err}".format(err=err))
            sys.exit(-1)

    # Find top K key phrase similar to the paper
    # Ref: https://www.sbert.net/examples/applications/semantic-search/README.html
    @staticmethod
    def compute_similar_score_key_phrases(model, doc_text, candidates):
        try:
            if len(candidates) == 0:
                return []

            # Encode cluster doc and keyword candidates into vectors for comparing the similarity
            candidate_vectors = model.encode(candidates, convert_to_numpy=True)
            doc_vector = model.encode([doc_text], convert_to_numpy=True)  # Convert the numpy array
            # Compute the distance of doc vector and each candidate vector
            distances = cosine_similarity(doc_vector, candidate_vectors)[0].tolist()
            # Select top key phrases based on the distance score
            candidate_scores = list()
            # Get all the candidates sorted by similar score
            for candidate, distance in zip(candidates, distances):
                found = next((kp for kp in candidate_scores if kp['key-phrase'].lower() == candidate.lower()), None)
                if not found:
                    candidate_scores.append({'key-phrase': candidate, 'score': distance})
            # Sort the phrases by scores
            candidate_scores = sorted(candidate_scores, key=lambda k: k['score'], reverse=True)
            return candidate_scores
        except Exception as err:
            print("Error occurred! {err}".format(err=err))

    # Get a list of unique key phrases from all papers
    @staticmethod
    def sort_phrases_by_similar_score(phrase_scores):
        try:
            # Sort 'phrase list'
            sorted_phrase_list = sorted(phrase_scores, key=lambda p: p['score'], reverse=True)
            unique_key_phrases = list()
            for key_phrase in sorted_phrase_list:
                # find if key phrase exist in all key phrase list
                found = next((kp for kp in unique_key_phrases
                              if kp['key-phrase'].lower() == key_phrase['key-phrase'].lower()), None)
                if not found:
                    unique_key_phrases.append(key_phrase)
                else:
                    print("Duplicated: " + found['key-phrase'])

            # Return unique key phrases
            return unique_key_phrases
        except Exception as _err:
            print("Error occurred! {err}".format(err=_err))

    @staticmethod
    def group_cluster_key_phrases_with_opt_parameter(parameter, doc_key_phrases):
        # Collect the key phrases linked to the docs
        def _collect_doc_ids(_doc_key_phrases, _grouped_key_phrases):
            _doc_ids = list()
            for _doc in _doc_key_phrases:
                # find if the doc contains any key phrases in its top 5 phrase list
                for _candidate in _doc['Key-phrases']:
                    _found = next((_gkp for _gkp in _grouped_key_phrases if _gkp.lower() == _candidate.lower()), None)
                    if _found:
                        _doc_ids.append(_doc['DocId'])
                        break
            return _doc_ids

        try:
            # Aggregate all the key phrases of each doc in a cluster as a single list
            key_phrases = reduce(lambda pre, cur: pre + cur['Key-phrases'], doc_key_phrases, list())
            # Filter duplicate key phrases
            unique_key_phrase = list()
            for key_phrase in key_phrases:
                found = next((k for k in unique_key_phrase if k.lower() == key_phrase.lower()), None)
                if not found:
                    unique_key_phrase.append(key_phrase)
            # Get the grouping labels of key phrases
            group_labels = parameter['group_labels']
            # Load key phrase and group labels
            df = pd.DataFrame()
            df['Key-phrases'] = unique_key_phrase
            df['Group'] = group_labels
            # Output the summary of the grouped key phrase results
            group_df = df.groupby(by=['Group'], as_index=False).agg({'Key-phrases': lambda k: list(k)})
            # Output the summary results to a csv file
            group_df['NumPhrases'] = group_df['Key-phrases'].apply(len)
            # Collect doc ids that contained the grouped key phrases
            group_key_phrases = group_df['Key-phrases'].tolist()
            group_df['DocIds'] = list(map(lambda group: _collect_doc_ids(doc_key_phrases, group), group_key_phrases))
            group_df['NumDocs'] = group_df['DocIds'].apply(len)
            return group_df.to_dict("records")
        except Exception as err:
            print("Error occurred! {err}".format(err=err))

    @staticmethod
    def group_key_phrases_with_opt_parameter(parameter, key_phrases, doc_key_phrases):
        # Collect the key phrases linked to the docs
        def _collect_doc_ids(_doc_key_phrases, _group):
            _doc_ids = list()
            for _doc in _doc_key_phrases:
                # find if any doc key phrases appear in the grouped key phrases
                for _candidate in _doc['Key-phrases']:
                    _found = next((_key_phrase for _key_phrase in _group if _key_phrase.lower() == _candidate.lower()),
                                  None)
                    if _found:
                        _doc_ids.append(_doc['DocId'])
                        break
            return _doc_ids

        try:
            # Get the grouping labels of key phrases
            group_labels = parameter['group_labels']
            # Load key phrase and group labels
            df = pd.DataFrame()
            df['Key-phrases'] = key_phrases
            df['Group'] = group_labels
            # Output the summary of the grouped key phrase results
            group_df = df.groupby(by=['Group'], as_index=False).agg({'Key-phrases': lambda k: list(k)})
            # Output the summary results to a csv file
            group_df['NumPhrases'] = group_df['Key-phrases'].apply(len)
            # Collect doc ids that contained the grouped key phrases
            group_key_phrases = group_df['Key-phrases'].tolist()
            group_df['DocIds'] = list(map(lambda group: _collect_doc_ids(doc_key_phrases, group), group_key_phrases))
            group_df['NumDocs'] = group_df['DocIds'].apply(len)
            return group_df.to_dict("records")
        except Exception as err:
            print("Error occurred! {err}".format(err=err))

    # Cluster key phrases (vectors) using HDBSCAN clustering
    @staticmethod
    def group_key_phrase_experiments_by_HDBSCAN(key_phrases, model, n_neighbors=100):
        def collect_group_results(_results, _group_label):
            try:
                _found = next((r for r in _results if r['group'] == _group_label), None)
                if not _found:
                    _found = {'group': _group_label, 'count': 1}
                    _results.append(_found)
                else:
                    _found['count'] += 1
                # Sort the results
                _results = sorted(_results, key=lambda c: (c['count'], c['group']))
                return _results
            except Exception as _err:
                print("Error occurred! {err}".format(err=_err))

        dimensions = [100, 90, 80, 70, 60, 50, 40, 30, 20, 10, 5]
        min_sample_list = [20, 15, 10, 5, 1]
        min_cluster_size_list = list(range(30, 10, -1))
        # min_cluster_size_list = list(range(30, 20, -1))
        try:
            # Convert the key phrases to vectors
            key_phrase_vectors = model.encode(key_phrases)
            vector_list = key_phrase_vectors.tolist()
            results = list()
            # Filter out dimensions > the length of key phrases
            dimensions = list(filter(lambda d: d < len(key_phrases) - 5, dimensions))
            for dimension in dimensions:
                # Reduce the doc vectors to specific dimension
                reduced_vectors = umap.UMAP(
                    n_neighbors=n_neighbors,
                    min_dist=0.0,
                    n_components=dimension,
                    random_state=42,
                    metric="cosine").fit_transform(vector_list)
                # Get vector dimension
                # Compute the cosine distance/similarity for each doc vectors
                distances = pairwise_distances(reduced_vectors, metric='cosine')
                for min_samples in min_sample_list:
                    for min_cluster_size in min_cluster_size_list:
                        for epsilon in [0.0]:
                            try:
                                # Group key phrase vectors using HDBSCAN clustering
                                group_labels = hdbscan.HDBSCAN(min_cluster_size=min_cluster_size,
                                                               min_samples=min_samples,
                                                               cluster_selection_epsilon=epsilon,
                                                               metric='precomputed').fit_predict(
                                    distances.astype('float64')).tolist()
                                group_results = reduce(lambda pre, cur: collect_group_results(pre, cur),
                                                       group_labels, list())
                                outlier_number = next((g['count'] for g in group_results if g['group'] == -1), 0)
                                if len(group_results) > 1:
                                    df = pd.DataFrame()
                                    df['Group'] = group_labels
                                    df['Vector'] = distances.tolist()
                                    # Remove the outliers
                                    no_outlier_df = df[df['Group'] != -1]
                                    no_outlier_labels = no_outlier_df['Group'].tolist()
                                    no_outlier_vectors = np.vstack(no_outlier_df['Vector'].tolist())
                                    score = BERTModelDocClusterUtility.compute_Silhouette_score(no_outlier_labels,
                                                                                                no_outlier_vectors)
                                else:  # All key phrases are identified as outliers
                                    score = -1
                                # Output the result
                                result = {'dimension': dimension,
                                          'min_samples': str(min_samples),
                                          'min_cluster_size': min_cluster_size,
                                          'epsilon': epsilon,
                                          'total_groups': len(group_results),
                                          'outliers': outlier_number,
                                          'score': round(score, 4),
                                          'group_results': group_results,
                                          'group_labels': group_labels}
                                results.append(result)
                                # print(result)
                            except Exception as err:
                                print("Error occurred! {err}".format(err=err))
                print("=== Complete to group the key phrases at dimension {d} ===".format(d=dimension))
            # Return all experiment results
            return results
        except Exception as err:
            print("Error occurred! {err}".format(err=err))
            sys.exit(-1)

    # # Compute the RAKE score
    # # Ref: https://github.com/zelandiya/RAKE-tutorial
    @staticmethod
    def rank_key_phrases_by_rake_scores(phrase_list):
        # Compute the score for a single word
        def _calculate_rake_word_scores(_phraseList):
            _word_frequency = {}
            _word_degree = {}
            for _phrase in _phraseList:
                _word_list = word_tokenize(_phrase.lower())
                _word_list_length = len(_word_list)
                _word_list_degree = _word_list_length - 1
                # if word_list_degree > 3: word_list_degree = 3 #exp.
                for _word in _word_list:
                    _word_frequency.setdefault(_word, 0)
                    _word_frequency[_word] += 1
                    _word_degree.setdefault(_word, 0)
                    _word_degree[_word] += _word_list_degree  # orig.
            # Calculate the word degree
            for _word in _word_frequency:
                _word_degree[_word] = _word_degree[_word] + _word_frequency[_word]

            # Calculate Word scores = deg(w)/freq(w)
            _word_scores = {}
            for _word in _word_frequency:
                _word_scores.setdefault(_word, 0)
                _word_scores[_word] = _word_degree[_word] / (_word_frequency[_word] * 1.0)  # orig.
            return _word_scores

        try:
            # Compute the word scores
            word_scores = _calculate_rake_word_scores(phrase_list)
            keyword_scores_dict = {}
            for phrase in phrase_list:
                keyword_scores_dict.setdefault(phrase, 0)
                word_list = word_tokenize(phrase.lower())
                candidate_score = 0
                for word in word_list:
                    candidate_score += word_scores[word]
                keyword_scores_dict[phrase] = candidate_score
            # Convert dict (keyword_scores_dict)
            keyword_scores = list()
            for keyword, score in keyword_scores_dict.items():
                keyword_scores.append({"key-phrase": keyword, "score": score})
            keyword_scores = sorted(keyword_scores, key=lambda ks: ks['score'], reverse=True)
            return keyword_scores
        except Exception as err:
            print("Error occurred! {err}".format(err=err))

    # Sort the key phrases by counting the occurrences of frequent words.
    # A phrase contains a more number of freq words and has a high rank
    @staticmethod
    def rank_key_phrases_by_top_word_freq(freq_words, key_phrases):
        key_phrase_scores = list()
        for key_phrase in key_phrases:
            try:
                # Check if the key phrase contains any freq_words
                found_words = [w for w in freq_words if w.lower() in key_phrase]
                key_phrase_scores.append({"key-phrase": key_phrase, "score": len(found_words)})
            except Exception as err:
                print("Error occurred! {err}".format(err=err))
                sys.exit(-1)
        # Sort the list by score
        sorted_key_phrases = sorted(key_phrase_scores, key=lambda ks: (ks['score'], ks['key-phrase'].lower()),
                                    reverse=True)
        return list(map(lambda ks: ks['key-phrase'], sorted_key_phrases))

    # Maximal Marginal Relevance minimizes redundancy and maximizes the diversity of results
    # Ref: https://towardsdatascience.com/keyword-extraction-with-bert-724efca412ea
    @staticmethod
    def re_rank_phrases_by_maximal_margin_relevance(model, doc_text, phrase_candidates, diversity=0.0, top_k=20):
        try:
            top_n = min(20, len(phrase_candidates))
            doc_vector = model.encode([doc_text], convert_to_numpy=True)
            phrase_vectors = model.encode(phrase_candidates, show_progress_bar=True, convert_to_numpy=True)

            # Extract similarity within words, and between words and the document
            phrase_doc_similarity = cosine_similarity(phrase_vectors, doc_vector)
            phrase_similarity = cosine_similarity(phrase_vectors, phrase_vectors)

            # Pick up the most similar phrase
            most_similar_index = np.argmax(phrase_doc_similarity)
            # Initialize candidates and already choose best keyword/key phrases
            key_phrase_idx = [most_similar_index]
            top_phrases = [{'key-phrase': (phrase_candidates[most_similar_index]),
                            'score': phrase_doc_similarity[most_similar_index][0]}]
            # Get all the remaining index
            candidate_indexes = list(filter(lambda idx: idx != most_similar_index, range(len(phrase_candidates))))
            # Add the other candidate phrase
            for i in range(0, top_n - 1):
                # Get similarities between doc and candidates
                candidate_similarities = phrase_doc_similarity[candidate_indexes, :]
                # Get similarity between candidates and a set of extracted key phrases
                target_similarities = phrase_similarity[candidate_indexes][:, key_phrase_idx]
                # Calculate MMR
                mmr_scores = (1 - diversity) * candidate_similarities - diversity * np.max(target_similarities,
                                                                                           axis=1).reshape(-1, 1)
                mmr_idx = candidate_indexes[np.argmax(mmr_scores)]

                # Update keywords & candidates
                top_phrases.append(
                    {'key-phrase': phrase_candidates[mmr_idx], 'score': phrase_doc_similarity[mmr_idx][0]})
                key_phrase_idx.append(mmr_idx)
                # Remove the phrase at mmr_idx from candidate
                candidate_indexes = list(filter(lambda idx: idx != mmr_idx, candidate_indexes))
            return top_phrases
        except Exception as err:
            print("Error occurred! {err}".format(err=err))
            sys.exit(-1)

    # Run HDBSCAN experiments to re-group the phrases at 'i' iteration
    @staticmethod
    def run_re_grouping_experiments(cluster_no, model, key_phrase_groups, doc_key_phrases):
        # Collect the key phrases linked to the docs
        def _collect_doc_ids(_doc_key_phrases, _group):
            _doc_ids = list()
            for _doc in _doc_key_phrases:
                # find if any doc key phrases appear in the grouped key phrases
                for _candidate in _doc['Key-phrases']:
                    _found = next((_key_phrase for _key_phrase in _group if _key_phrase.lower() == _candidate.lower()),
                                  None)
                    if _found:
                        _doc_ids.append(_doc['DocId'])
                        break
            return _doc_ids

        try:

            # Re-group the group size > 30 only
            key_phrase_groups = list(filter(lambda g: len(g['Key-phrases']) > 30, key_phrase_groups))
            # Store experiment results
            results = list()
            cur_group_no = 1
            # Run the grouping experiments to regroup the key phrases
            for group in key_phrase_groups:
                try:

                    key_phrases = group['Key-phrases']
                    if len(key_phrases) < 40:
                        group['Group'] = cur_group_no
                        results.append(group)
                        cur_group_no = cur_group_no + 1
                    else:
                        experiments = KeyPhraseUtility.group_key_phrase_experiments_by_HDBSCAN(key_phrases, model)
                        # Sort the experiments by sort
                        experiments = sorted(experiments, key=lambda ex: (ex['score'], ex['dimension']),
                                             reverse=True)
                        # Get the best experiment
                        best_ex = experiments[0]
                        # Get the grouping labels of key phrases
                        group_labels = best_ex['group_labels']
                        group_list = list(set(group_labels))
                        if len(group_list) > 1:
                            grouping_results = list(zip(key_phrases, group_labels))
                            for group_no in group_list:
                                sub_key_phrases = list(map(lambda g: g[0], list(filter(lambda g: g[1] == group_no, grouping_results))))
                                print(sub_key_phrases)
                        else:
                            group['Group'] = cur_group_no
                            results.append(group)
                            cur_group_no = cur_group_no + 1
                        # print(grouping_results)
                except Exception as err:
                    print("Error occurred! {err}".format(err=err))
                    sys.exit(-1)
            print("=== Complete grouping the key phrases of cluster {no} ===".format(no=cluster_no))
            return results
        except Exception as err:
            print("Error occurred! {err}".format(err=err))
            sys.exit(-1)

    # Obtain the top 3 frequent words from a list of phrases
    @staticmethod
    def get_top_frequent_words(phrases):
        try:
            freq_word_dict = {}
            # Count the word frequencies
            for phrase in phrases:
                words = phrase.split(" ")
                for word in words:
                    upper_cases = re.findall(r'[A-Z]', word)
                    # Check if the word contain all upper cases of chars or at least two chars
                    if word.isupper() or len(upper_cases) >= 2:
                        # Keep the cases of the word
                        freq_word_dict.setdefault(word, 0)
                        freq_word_dict[word] += 1
                    else:
                        freq_word_dict.setdefault(word.lower(), 0)
                        freq_word_dict[word.lower()] += 1
            # Convert the dict to list
            freq_words = list()
            for word, freq in freq_word_dict.items():
                freq_words.append({'word': word, 'freq': freq})
            # Sort the list
            freq_words = sorted(freq_words, key=lambda fw: fw['freq'], reverse=True)
            # assert len(freq_words) >= 3
            top_words = list(map(lambda fw: fw['word'], freq_words))
            return top_words[:3]
        except Exception as err:
            print("Error occurred! {err}".format(err=err))
            sys.exit(-1)

