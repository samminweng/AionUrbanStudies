# Helper function for cluster Similarity
import re
import string
from functools import reduce
from pathlib import Path

import pandas as pd
from coverage.annotate import os
from nltk import sent_tokenize, word_tokenize, pos_tag, ngrams

from BERTModelDocClusterUtility import BERTModelDocClusterUtility


class ClusterTopicUtility:
    @staticmethod
    def collect_iterative_cluster_topic_results(case_name, last_iteration):
        cluster_folder = os.path.join('output', case_name, 'cluster')
        results = list()
        # Go through each iteration 1 to last iteration
        for i in range(0, last_iteration + 1):
            try:
                dimension = 0
                # Get the best dimension
                folder = os.path.join(cluster_folder, 'iteration_' + str(i), 'hdbscan_clustering')
                for file in os.listdir(folder):
                    file_name = file.lower()
                    if file_name.endswith(".png") and file_name.startswith("dimension"):
                        dimension = int(file_name.split("_")[1].split(".png")[0])
                # Get the best score
                folder = os.path.join(cluster_folder, 'iteration_' + str(i), 'experiments')
                path = os.path.join(folder, 'HDBSCAN_cluster_doc_vector_result_summary.json')
                experiment_results = pd.read_json(path).to_dict("records")
                best_result = next(r for r in experiment_results if r['dimension'] == dimension)
                min_samples = best_result['min_samples']
                min_cluster_size = best_result['min_cluster_size']
                score = best_result['Silhouette_score']
                # Get summary of cluster topics
                folder = os.path.join(cluster_folder, 'iteration_' + str(i), 'topics')
                path = os.path.join(folder, 'TF-IDF_cluster_topic_summary.json')
                cluster_topics = pd.read_json(path).to_dict("records")
                total_papers = reduce(lambda total, ct: ct['NumDocs'] + total, cluster_topics, 0)
                for ct in cluster_topics:
                    results.append({
                        "iteration": i, "total_papers": total_papers, "dimension": dimension,
                        "min_samples": min_samples, "min_cluster_size": min_cluster_size, "score": score,
                        "Cluster": ct['Cluster'], "NumDocs": ct['NumDocs'], "Percent": ct['Percent'],
                        "DocIds": ct['DocIds']
                    })
            except Exception as _err:
                print("Error occurred! {err}".format(err=_err))
        # Load the results as data frame
        df = pd.DataFrame(results)
        # Output cluster results to CSV
        folder = os.path.join('output', case_name, 'cluster')
        path = os.path.join(folder, case_name + '_iterative_cluster_topic_summary.csv')
        df.to_csv(path, encoding='utf-8', index=False)
        path = os.path.join(folder, case_name + '_iterative_cluster_topic_summary.json')
        df.to_json(path, orient='records')
        print(df)

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