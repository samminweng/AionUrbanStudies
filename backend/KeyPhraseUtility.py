import csv
import getpass
import itertools
import os
import re
import string
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
from sentence_transformers import util
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


# Load the lemma.n file to store the mapping of singular to plural nouns
def load_lemma_nouns():
    _lemma_nouns = {}
    path = os.path.join('data', 'lemma.n')
    f = open(path, 'r')
    lines = f.readlines()
    for line in lines:
        words = line.rstrip().split("->")  # Remove trailing new line char and split by '->'
        plural_word = words[1]
        if '.,' in plural_word:  # Handle multiple plural forms and get the last one as default plural form
            plural_word = plural_word.split('.,')[-1]
        singular_word = words[0]
        _lemma_nouns[plural_word] = singular_word
    return _lemma_nouns


lemma_nouns = load_lemma_nouns()


# Helper function for cluster Similarity
class KeyPhraseUtility:
    stop_words = list(stopwords.words('english'))

    # Clean licence statement texts
    @staticmethod
    def clean_sentence(text):
        # Split the words 'within/near'
        def split_words(_words):
            _out_words = list()
            for _word in _words:
                if matches := re.match(r'(\w+)/(\w+)', _word):
                    _out_words.append(matches.group(1))
                    _out_words.append(matches.group(2))
                elif re.match(r"('\w+)|(\w+')", _word):
                    _out_words.append(_word.replace("'", ""))
                else:
                    _out_words.append(_word)
            return _out_words

        # Change plural nouns to singular nouns using lemmatizer
        def convert_singular_words(_words, _lemmatiser):
            # Tag the words with part-of-speech tags
            _pos_tags = nltk.pos_tag(_words)
            # Convert plural word to singular
            singular_words = []
            for i, (_word, _pos_tag) in enumerate(_pos_tags):
                try:
                    # Lowercase 1st char of the firs word
                    # if i == 0:
                    #    _word = _word[0].lower() + _word[1:len(_word)]
                    # NNS indicates plural nouns and convert the plural noun to singular noun
                    if _pos_tag == 'NNS':
                        singular_word = _lemmatiser.lemmatize(_word.lower())
                        if _word[0].isupper():  # Restore the uppercase
                            singular_word = singular_word.capitalize()  # Upper case the first character
                        singular_words.append(singular_word)
                    else:
                        # Check if the word in lemma list
                        if _word.lower() in lemma_nouns:
                            try:
                                singular_word = lemma_nouns[_word.lower()]
                                if _word[0].isupper():  # Restore the uppercase
                                    singular_word = singular_word.capitalize()  # Upper case the first character
                                singular_words.append(singular_word)
                            except Exception as _err:
                                print("Error occurred! {err}".format(err=_err))
                        else:
                            singular_words.append(_word)
                except Exception as _err:
                    print("Error occurred! {err}".format(err=_err))
            # Return all lemmatized words
            return singular_words

        lemmatizer = WordNetLemmatizer()
        sentences = sent_tokenize(text)
        # Preprocess the sentence
        cleaned_sentences = list()  # Skip copy right sentence
        for sentence in sentences:
            if u"\u00A9" not in sentence.lower() and 'licensee' not in sentence.lower() \
                    and 'copyright' not in sentence.lower() and 'rights reserved' not in sentence.lower():
                try:
                    words = split_words(word_tokenize(sentence))
                    if len(words) > 0:
                        # Convert the plural words into singular words
                        cleaned_words = convert_singular_words(words, lemmatizer)
                        cleaned_sentences.append(cleaned_words)  # merge tokenized words into sentence
                except Exception as err:
                    print("Error occurred! {err}".format(err=err))
        return cleaned_sentences  # Convert a list of clean sentences to the text

    @staticmethod
    # Generate n-gram of a text and avoid stop
    def generate_n_gram_candidates(sentences, n_gram_range):
        def _is_qualified(_n_gram):  # _n_gram is a list of tuple (word, tuple)
            try:
                qualified_tags = ['NN', 'NNS', 'JJ', 'NNP']
                # # Check if there is any noun
                nouns = list(filter(lambda _n: _n[1].startswith('NN'), _n_gram))
                if len(nouns) == 0:
                    return False
                # Check the last word is a nn or nns
                if _n_gram[-1][1] not in ['NN', 'NNS']:
                    return False
                # Check if all words are not stop word or punctuation or non-words
                for _i, _n in enumerate(_n_gram):
                    _word = _n[0]
                    _pos_tag = _n[1]
                    if bool(re.search(r'\d|[^\w]', _word.lower())) or _word.lower() in string.punctuation or \
                            _word.lower() in KeyPhraseUtility.stop_words or _pos_tag not in qualified_tags:
                        return False
                # n-gram is qualified
                return True
            except Exception as err:
                print("Error occurred! {err}".format(err=err))

        candidates = list()
        # Extract n_gram from each sentence
        for i, sentence in enumerate(sentences):
            try:
                pos_tags = pos_tag(sentence)
                # Pass pos tag tuple (word, pos-tag) of each word in the sentence to produce n-grams
                n_grams = list(ngrams(pos_tags, n_gram_range))
                # Filter out not qualified n_grams that contain stopwords or the word is not alpha_numeric
                for n_gram in n_grams:
                    if _is_qualified(n_gram):
                        n_gram_text = " ".join(list(map(lambda n: n[0], n_gram)))
                        # Check if candidates exist in the list
                        found = next((c for c in candidates if c.lower() == n_gram_text.lower()), None)
                        if not found:
                            candidates.append(n_gram_text)  # Convert n_gram (a list of words) to a string
            except Exception as err:
                print("Error occurred! {err}".format(err=err))
        return candidates

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
    def get_top_similar_key_phrases(phrase_list, top_k=5):
        try:
            if len(phrase_list) < top_k:
                return phrase_list

            # Sort 'phrase list'
            sorted_phrase_list = sorted(phrase_list, key=lambda p: p['score'], reverse=True)
            unique_key_phrases = list()
            for key_phrase in sorted_phrase_list:
                # find if key phrase exist in all key phrase list
                found = next((kp for kp in unique_key_phrases
                              if kp['key-phrase'].lower() == key_phrase['key-phrase'].lower()), None)
                if not found:
                    unique_key_phrases.append(key_phrase)
                else:
                    print("Duplicated: " + found['key-phrase'])

            # Get top 5 key phrase
            unique_key_phrases = unique_key_phrases[:top_k]
            # assert len(_unique_key_phrases) == _top_k
            return list(map(lambda p: p['key-phrase'], unique_key_phrases))
        except Exception as _err:
            print("Error occurred! {err}".format(err=_err))

    @staticmethod
    def group_key_phrases_with_best_result(cluster_no, parameter, doc_key_phrases, folder):
        # Collect the key phrases linked to the docs
        def get_doc_ids_by_group_key_phrases(_doc_key_phrases, _grouped_key_phrases):
            _doc_ids = list()
            for doc in _doc_key_phrases:
                # find if any doc key phrases appear in the grouped key phrases
                for key_phrase in doc['key-phrases']:
                    found = next((gkp for gkp in _grouped_key_phrases if gkp.lower() == key_phrase.lower()), None)
                    if found:
                        _doc_ids.append(doc['DocId'])
                        break
            return _doc_ids

        try:
            # Aggregate all the key phrases of each doc in a cluster as a single list
            key_phrases = reduce(lambda pre, cur: pre + cur['key-phrases'], doc_key_phrases, list())
            # Get the grouping labels of key phrases
            group_labels = parameter['group_labels']
            # Load key phrase and group labels
            df = pd.DataFrame()
            df['key-phrases'] = key_phrases
            df['group'] = group_labels
            # Output the summary of the grouped key phrase results
            group_df = df.groupby(by=['group'], as_index=False).agg({'key-phrases': lambda k: list(k)})
            # Output the summary results to a csv file
            group_df['count'] = group_df['key-phrases'].apply(len)
            # Collect doc ids that contained the grouped key phrases
            group_key_phrases = group_df['key-phrases'].tolist()
            group_doc_ids = list(
                map(lambda group: get_doc_ids_by_group_key_phrases(doc_key_phrases, group), group_key_phrases))
            group_df['DocIds'] = group_doc_ids
            group_df['NumDocs'] = group_df['DocIds'].apply(len)
            group_df = group_df[['group', 'count', 'key-phrases', 'NumDocs', 'DocIds']]  # Re-order the column list
            path = os.path.join(folder, 'top_key_phrases_cluster_#' + str(cluster_no) + '_best_grouping.csv')
            group_df.to_csv(path, encoding='utf-8', index=False)
            # Output the summary of best grouped key phrases to a json file
            path = os.path.join(folder, 'top_key_phrases_cluster_#' + str(cluster_no) + '_best_grouping.json')
            group_df.to_json(path, orient='records')
            print('Output the summary of grouped key phrase to ' + path)
            return group_df.to_dict("records")
        except Exception as err:
            print("Error occurred! {err}".format(err=err))

    # Cluster key phrases (vectors) using HDBSCAN clustering
    @staticmethod
    def group_key_phrase_experiments_by_HDBSCAN(key_phrases, cluster_no, model, folder, n_neighbors=5):
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

        try:
            # Convert the key phrases to vectors
            key_phrase_vectors = model.encode(key_phrases)
            vector_list = key_phrase_vectors.tolist()
            results = list()
            dimensions = [100, 90, 80, 70, 60, 50, 40, 30, 20, 15, 10, 5]
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
                for min_samples in [None] + list(range(1, 21)):
                    for min_cluster_size in list(range(5, 21)):
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
                                    df['groups'] = group_labels
                                    df['vectors'] = distances.tolist()
                                    # Remove the outliers
                                    no_outlier_df = df[df['groups'] != -1]
                                    no_outlier_labels = no_outlier_df['groups'].tolist()
                                    no_outlier_vectors = np.vstack(no_outlier_df['vectors'].tolist())
                                    score = BERTModelDocClusterUtility.compute_Silhouette_score(no_outlier_labels,
                                                                                                no_outlier_vectors)
                                else:  # All key phrases are identified as outliers
                                    score = -1
                                # Output the result
                                result = {'cluster': "#" + str(cluster_no),
                                          'dimension': dimension,
                                          'min_samples': str(min_samples),
                                          'min_cluster_size': min_cluster_size,
                                          'epsilon': epsilon,
                                          'total_groups': len(group_results),
                                          'outliers': outlier_number,
                                          'score': score,
                                          'group_result': group_results,
                                          'group_labels': group_labels}
                                results.append(result)
                                # print(result)
                            except Exception as err:
                                print("Error occurred! {err}".format(err=err))
                print(
                    "=== Complete grouping the key phrases of cluster {no} with dimension {d} ===".format(no=cluster_no,
                                                                                                          d=dimension))
            # output the experiment results
            df = pd.DataFrame(results)
            path = os.path.join(folder, 'top_key_phrases_cluster_#' + str(cluster_no) + '_grouping_experiments.csv')
            df.to_csv(path, encoding='utf-8', index=False)
            path = os.path.join(folder, 'top_key_phrases_cluster_#' + str(cluster_no) + '_grouping_experiments.json')
            df.to_json(path, orient='records')
        except Exception as err:
            print("Error occurred! {err}".format(err=err))

    # # Compute the RAKE score
    # # Ref: https://github.com/zelandiya/RAKE-tutorial
    @staticmethod
    def compute_keyword_rake_scores(phrase_list):
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
