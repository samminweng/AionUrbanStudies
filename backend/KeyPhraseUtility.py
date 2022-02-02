import csv
import getpass
import itertools
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
                        # # Check if the word in lemma list
                        # if _word.lower() in lemma_nouns:
                        #     try:
                        #         singular_word = lemma_nouns[_word.lower()]
                        #         if _word[0].isupper():  # Restore the uppercase
                        #             singular_word = singular_word.capitalize()  # Upper case the first character
                        #         singular_words.append(singular_word)
                        #     except Exception as _err:
                        #         print("Error occurred! {err}".format(err=_err))
                        # else:
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
                # For debugging
                # if 'contains' in sentence and 'mobility' in sentence:
                #     print("bug")
                #     print(" ".join(sentence))
                #     print(pos_tags)
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
                # find if any doc key phrases appear in the grouped key phrases
                for _candidate in _doc['Phrase-candidates']:
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
                for _candidate in _doc['Phrase-candidates']:
                    _found = next((_key_phrase for _key_phrase in _group if _key_phrase.lower() == _candidate.lower()), None)
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
    def group_key_phrase_experiments_by_HDBSCAN(key_phrases, model, n_neighbors=3):
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
            dimensions = [100, 95, 90, 85, 80, 75, 70, 65, 60, 55, 50, 45, 40, 35, 30, 25, 20, 15, 10, 5]
            # dimensions = [100, 50]
            # Filter out dimensions > the length of key phrases
            dimensions = list(filter(lambda d: d < len(key_phrases) - 5, dimensions))
            min_cluster_size_list = list(range(20, 4, -1))
            min_sample_list = [50, 45, 40, 35, 30, 25, 20, 15, 10, 5]
            if (len(key_phrases)/5) > 30:
                min_cluster_size_list = [60, 55, 50, 45, 40, 35, 30, 25, 20, 15]

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
            for i in range(0, top_n-1):
                # Get similarities between doc and candidates
                candidate_similarities = phrase_doc_similarity[candidate_indexes, :]
                # Get similarity between candidates and a set of extracted key phrases
                target_similarities = phrase_similarity[candidate_indexes][:, key_phrase_idx]
                # Calculate MMR
                mmr_scores = (1 - diversity) * candidate_similarities - diversity * np.max(target_similarities, axis=1).reshape(-1, 1)
                mmr_idx = candidate_indexes[np.argmax(mmr_scores)]

                # Update keywords & candidates
                top_phrases.append({'key-phrase': phrase_candidates[mmr_idx], 'score': phrase_doc_similarity[mmr_idx][0]})
                key_phrase_idx.append(mmr_idx)
                # Remove the phrase at mmr_idx from candidate
                candidate_indexes = list(filter(lambda idx: idx != mmr_idx, candidate_indexes))
            return top_phrases
        except Exception as err:
            print("Error occurred! {err}".format(err=err))
            sys.exit(-1)