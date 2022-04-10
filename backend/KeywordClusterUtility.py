import getpass
import math
import os
import re
import string
import sys
from functools import reduce
import hdbscan
import nltk
import umap
from nltk import word_tokenize, sent_tokenize, ngrams, pos_tag
import pandas as pd
import numpy as np
# Load function words
from nltk.corpus import stopwords
from sklearn.metrics.pairwise import cosine_similarity, pairwise_distances
from BERTArticleClusterUtility import BERTArticleClusterUtility

nltk_path = os.path.join('/Scratch', getpass.getuser(), 'nltk_data')
if os.name == 'nt':
    nltk_path = os.path.join("C:", os.sep, "Users", getpass.getuser(), "nltk_data")
nltk.download('punkt', download_dir=nltk_path)
nltk.download('wordnet', download_dir=nltk_path)
nltk.download('stopwords', download_dir=nltk_path)
nltk.download('averaged_perceptron_tagger', download_dir=nltk_path)  # POS tags
# Append NTLK data path
nltk.data.path.append(nltk_path)


# Helper function for keyword cluster
class KeywordClusterUtility:
    stop_words = list(stopwords.words('english'))

    @staticmethod
    def write_keyword_cluster_summary(results, folder):
        # Write the results to a summary
        kp_group_summary = list()
        for result in results:
            kp_groups = result['Key-phrases']
            score = kp_groups[0]['score']
            summary = {"cluster": result['Cluster'], "count": len(kp_groups), "score": score}
            total = 0
            for group_no in range(1, 6):
                if group_no <= len(kp_groups):
                    num_phrases = kp_groups[group_no - 1]['NumPhrases']
                    summary['kp_cluster#' + str(group_no)] = num_phrases
                    total = total + num_phrases
            summary['total'] = total
            kp_group_summary.append(summary)
        # Write keyword group results to a summary (csv)
        path = os.path.join(folder, "key_phrase_groups.csv")
        kp_group_df = pd.DataFrame(kp_group_summary, columns=['cluster', "count", "score", "total",
                                                              "kp_cluster#1", "kp_cluster#2", "kp_cluster#3",
                                                              "kp_cluster#4", "kp_cluster#5"])
        kp_group_df.to_csv(path, encoding='utf-8', index=False)

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
    def group_key_phrase_experiments_by_HDBSCAN(key_phrases, model, is_fined_grain=False, n_neighbors=40):
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

        dimensions = [100, 90, 80, 70, 60, 50, 45, 40, 35, 30]
        min_sample_list = [30, 25, 20, 15, 10]
        min_cluster_size_list = list(range(30, 14, -1))
        if is_fined_grain:
            dimensions = list(range(30, 9, -5))
            min_sample_list = list(range(10, 1, -1))
            min_cluster_size_list = list(range(30, 9, -1))
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
                                    score = BERTArticleClusterUtility.compute_Silhouette_score(no_outlier_labels,
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
                                sys.exit(-1)
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

    # Run HDBSCAN experiments to re-group the phrases at 'i' iteration
    @staticmethod
    def run_re_grouping_experiments(cluster_no, model, key_phrase_groups, doc_key_phrases):
        # Collect the key phrases linked to the docs
        def _collect_doc_ids(_doc_key_phrases, _key_phrases):
            _doc_ids = list()
            for _doc in _doc_key_phrases:
                # find if any doc key phrases appear in the grouped key phrases
                for _candidate in _doc['Key-phrases']:
                    _found = next((_key_phrase for _key_phrase in _key_phrases
                                   if _key_phrase.lower() == _candidate.lower()), None)
                    if _found:
                        _doc_ids.append(_doc['DocId'])
                        break
            return _doc_ids

        try:
            # Store experiment results
            results = list()
            cur_group_no = 1
            # Run the grouping experiments to regroup the key phrases
            for group in key_phrase_groups:
                try:
                    key_phrases = group['Key-phrases']
                    if len(key_phrases) < 30:
                        group['Group'] = cur_group_no
                        results.append(group)
                        cur_group_no = cur_group_no + 1
                    else:
                        experiments = KeywordClusterUtility.group_key_phrase_experiments_by_HDBSCAN(key_phrases, model,
                                                                                                    is_fined_grain=True)
                        # Sort the experiments by sort
                        experiments = sorted(experiments, key=lambda ex: (ex['score'], ex['min_cluster_size']),
                                             reverse=True)
                        # Get the best experiment
                        best_ex = experiments[0]
                        score = best_ex['score']
                        dimension = best_ex['dimension']
                        min_samples = best_ex['min_samples']
                        min_cluster_size = best_ex['min_cluster_size']
                        # Get the grouping labels of key phrases
                        group_labels = best_ex['group_labels']
                        group_list = list(set(group_labels))
                        if len(group_list) > 1:
                            grouping_results = list(zip(key_phrases, group_labels))
                            for group_no in group_list:
                                sub_key_phrases = list(
                                    map(lambda g: g[0], list(filter(lambda g: g[1] == group_no, grouping_results))))
                                # print(sub_key_phrases)
                                doc_ids = _collect_doc_ids(doc_key_phrases, sub_key_phrases)
                                new_group = {'Group': cur_group_no, 'NumPhrases': len(sub_key_phrases),
                                             'Key-phrases': sub_key_phrases,
                                             'DocIds': doc_ids, 'NumDocs': len(doc_ids),
                                             'score': score, 'dimension': dimension, 'min_samples': min_samples,
                                             'min_cluster_size': min_cluster_size}
                                results.append(new_group)
                                cur_group_no = cur_group_no + 1
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
