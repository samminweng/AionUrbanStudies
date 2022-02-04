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
    def group_key_phrase_experiments_by_HDBSCAN(key_phrases, min_cluster_size_list, model, n_neighbors=3):
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
            # min_cluster_size_list = list(range(30, 4, -1))
            min_sample_list = [50, 45, 40, 35, 30, 25, 20, 15, 10, 5]

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
    def run_re_grouping_experiments(level, cluster_no, key_phrase_folder, min_cluster_size_list, model, n_neighbors):
        try:
            # Load the best grouping of previous level
            path = os.path.join(key_phrase_folder, 'group_key_phrases', 'sub_groups', 'level_' + str(level - 1),
                                'group_key_phrases_cluster_#' + str(cluster_no) + '.json')
            key_phrase_groups = pd.read_json(path).to_dict("records")
            # Re-group the group size > 30 only
            key_phrase_groups = list(filter(lambda g: len(g['Key-phrases']) > 30, key_phrase_groups))
            # Store experiment results
            results = list()
            # Run the grouping experiments to regroup the key phrases
            for group in key_phrase_groups:
                try:
                    parent_group = group['Parent']
                    group_id = group['Group']
                    key_phrases = group['Key-phrases']
                    experiments = KeyPhraseUtility.group_key_phrase_experiments_by_HDBSCAN(key_phrases,
                                                                                           min_cluster_size_list,
                                                                                           model,
                                                                                           n_neighbors)
                    # Updated the parent group
                    for ex in experiments:
                        ex['parent_group'] = str(parent_group) + '_' + str(group_id)
                    results = results + experiments
                except Exception as err:
                    print("Error occurred! {err}".format(err=err))
                    sys.exit(-1)
            # Output all the experiment results
            folder = os.path.join(key_phrase_folder, 'experiments', 'level_' + str(level))
            Path(folder).mkdir(parents=True, exist_ok=True)
            # output the experiment results
            df = pd.DataFrame(results)
            path = os.path.join(folder,
                                'key_phrases_cluster_#' + str(cluster_no) + '_grouping_experiments.csv')
            df.to_csv(path, encoding='utf-8', index=False)
            path = os.path.join(folder,
                                'key_phrases_cluster_#' + str(cluster_no) + '_grouping_experiments.json')
            df.to_json(path, orient='records')
            print("=== Complete grouping the key phrases of cluster {no} ===".format(no=cluster_no))
        except Exception as err:
            print("Error occurred! {err}".format(err=err))
            sys.exit(-1)

    # Group the phrases based on the optimal results
    @staticmethod
    def re_group_phrases_by_opt_experiment(level, cluster_no, key_phrase_folder):
        try:
            # Load the key phrase groups at the previous iteration
            path = os.path.join(key_phrase_folder, 'group_key_phrases', 'sub_groups', 'level_' + str(level - 1),
                                'group_key_phrases_cluster_#' + str(cluster_no) + '.json')
            key_phrase_groups = pd.read_json(path).to_dict("records")
            # Re-group the group size > 30 only
            key_phrase_groups = list(filter(lambda g: len(g['Key-phrases']) > 30, key_phrase_groups))
            # Load the grouping experiment results
            ex_folder = os.path.join(key_phrase_folder, 'experiments', 'level_' + str(level))
            path = os.path.join(ex_folder,
                                'key_phrases_cluster_#' + str(cluster_no) + '_grouping_experiments.json')
            experiments = pd.read_json(path).to_dict("records")
            # Load key phrases of all the documents in a cluster
            path = os.path.join(key_phrase_folder, 'doc_key_phrase',
                                'doc_key_phrases_cluster_#{c}.json'.format(c=cluster_no))
            doc_key_phrases = pd.read_json(path).to_dict("records")
            results = list()
            # Load the experiment
            for group in key_phrase_groups:
                try:
                    parent_group = group['Parent'] + '_' + str(group['Group'])
                    key_phrases = group['Key-phrases']
                    group_experiments = list(filter(lambda ex: ex['parent_group'] == parent_group, experiments))
                    # # Sort the experiment results by score and dimension
                    group_experiments = sorted(group_experiments, key=lambda ex: (ex['score'], ex['dimension']),
                                               reverse=True)
                    # # Get the best results
                    opt_parameter = group_experiments[0]
                    # Obtain the grouped key phrases of the cluster
                    sub_group_key_phrases = KeyPhraseUtility.group_key_phrases_with_opt_parameter(opt_parameter,
                                                                                                  key_phrases,
                                                                                                  doc_key_phrases)
                    # Update the parent
                    for sub_group in sub_group_key_phrases:
                        try:
                            # Add the current group id as the parent of sub-group
                            sub_group['Parent'] = group['Parent'] + '_' + str(group['Group'])
                            sorted_phrase_scores = KeyPhraseUtility.rank_key_phrases_by_rake_scores(
                                sub_group['Key-phrases'])
                            sub_group['Key-phrases'] = list(map(lambda p: p['key-phrase'], sorted_phrase_scores))
                            sub_group['score'] = opt_parameter['score']
                            sub_group['dimension'] = opt_parameter['dimension']
                            sub_group['min_samples'] = opt_parameter['min_samples']
                            sub_group['min_cluster_size'] = opt_parameter['min_cluster_size']
                        except Exception as err:
                            print("Error occurred! {err}".format(err=err))
                            sys.exit(-1)
                    results = results + sub_group_key_phrases
                except Exception as err:
                    print("Error occurred! {err}".format(err=err))
                    sys.exit(-1)
            # # Output the grouped key phrases
            group_df = pd.DataFrame(results)
            group_df['Cluster'] = cluster_no
            group_df['NumPhrases'] = group_df['Key-phrases'].apply(len)
            group_df['NumDocs'] = group_df['DocIds'].apply(len)
            # Re-order the column list
            group_df = group_df[['Cluster', 'Parent', 'Group', 'NumPhrases', 'Key-phrases', 'NumDocs', 'DocIds',
                                 'score', 'dimension', 'min_samples', 'min_cluster_size']]
            folder = os.path.join(key_phrase_folder, 'group_key_phrases', 'sub_groups', 'level_' + str(level))
            Path(folder).mkdir(parents=True, exist_ok=True)
            path = os.path.join(folder, 'group_key_phrases_cluster_#' + str(cluster_no) + '.csv')
            group_df.to_csv(path, encoding='utf-8', index=False)
            path = os.path.join(folder, 'group_key_phrases_cluster_#' + str(cluster_no) + '.json')
            group_df.to_json(path, orient='records')
            print('Output the summary of grouped key phrase to ' + path)

            # Found if any subgroup > 30. If no subgroup > 30, then stop
            found = list(filter(lambda sg: len(sg['Key-phrases']) > 30, results))
            is_stop = False
            if len(found) == 0:
                is_stop = True
            return is_stop
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
            assert len(freq_words) >= 3
            top_words = list(map(lambda fw: fw['word'], freq_words))
            return top_words[:3]
        except Exception as err:
            print("Error occurred! {err}".format(err=err))
            sys.exit(-1)

    # Aggregate all the sub-groups to level 1
    @staticmethod
    def flat_key_phrase_subgroups(cluster_no, last_level, key_phrase_folder):
        all_sub_groups = list()
        # Collect all the groups and sub-groups
        for level in range(2, last_level):
            # Load parent level
            folder = os.path.join(key_phrase_folder, 'group_key_phrases', 'sub_groups', 'level_' + str(level))
            path = os.path.join(folder, 'group_key_phrases_cluster_#' + str(cluster_no) + '.json')
            sub_groups = pd.read_json(path).to_dict("records")
            # Find all the sub-groups starting with 'root_1'
            if level < (last_level - 1):
                sub_groups = list(filter(lambda g: len(g['Key-phrases']) <= 30, sub_groups))
            # Get the large groups
            all_sub_groups = all_sub_groups + sub_groups
        # Load the groups at level 1
        folder = os.path.join(key_phrase_folder, 'group_key_phrases', 'sub_groups', 'level_1')
        path = os.path.join(folder, 'group_key_phrases_cluster_#' + str(cluster_no) + '.json')
        # Get the large groups
        results = list()
        groups = pd.read_json(path).to_dict("records")
        for group in groups:
            parent = group['Parent']
            group_id = group['Group']
            group['TitleWords'] = KeyPhraseUtility.get_top_frequent_words(group['Key-phrases'])
            if len(group['Key-phrases']) <= 30:
                results.append(group)
            else:
                # Find all the sub-groups starting with 'root_1'
                sub_groups = list(filter(lambda g: g['Parent'].startswith(parent + '_' + str(group_id)), all_sub_groups))
                # Update the parents of sub-group
                for sub_group in sub_groups:
                    sub_group['Parent'] = parent
                    sub_group['TitleWords'] = KeyPhraseUtility.get_top_frequent_words(sub_group['Key-phrases'])
                results = results + sub_groups
        # Get all the unique parent ids
        parent_ids = list(set(map(lambda g: g['Parent'], results)))
        # Sort parent ids
        parent_ids = sorted(parent_ids, key=lambda p: int(p.split("_")[1]))
        sorted_results = list()
        for parent_id in parent_ids:
            sub_groups = list(filter(lambda g: g['Parent'] == parent_id, results))
            # Sort the sub groups by num phrases
            sub_groups = sorted(sub_groups, key=lambda g: int(g['NumPhrases']), reverse=True)
            # Re-number the sub-group
            g_id = 0
            for sub_group in sub_groups:
                sub_group['Group'] = g_id
                g_id += 1
            sorted_results = sorted_results + sub_groups
        # # print(best_results)
        # # Load best results of each group
        df = pd.DataFrame(sorted_results,
                          columns=['Cluster', 'Parent', 'Group', 'TitleWords', 'NumPhrases', 'Key-phrases',
                                   'NumDocs', 'DocIds'])
        df = df.rename(columns={'Parent': 'Group', 'Group': 'SubGroup'})
        df['Group'] = df['Group'].apply(lambda g: int(g.split("_")[1]))
        folder = os.path.join(key_phrase_folder, 'group_key_phrases', 'sub_groups')
        Path(folder).mkdir(parents=True, exist_ok=True)
        path = os.path.join(folder, 'cluster_key_phrases_sub_grouping_cluster_#' + str(cluster_no) + '.csv')
        df.to_csv(path, encoding='utf-8', index=False)
        path = os.path.join(folder, 'cluster_key_phrases_sub_grouping_cluster_#' + str(cluster_no) + '.json')
        df.to_json(path, orient="records")
        return df.to_dict("records")
