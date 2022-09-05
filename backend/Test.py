# Plots the chart to present our results in the papers
import copy
import getpass
import math
import os
import sys
from argparse import Namespace
from pathlib import Path
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

from KeyWordExtractionUtility import KeywordExtractionUtility


class Test:
    def __init__(self):
        self.args = Namespace(
            case_name='AIMLUrbanStudyCorpus',
            embedding_name='OpenAIEmbedding',
            model_name='curie',
            path='data'
        )
        # folder = os.path.join('output', self.args.case_name + '_' + self.args.embedding_name)
        # path = os.path.join(folder, self.args.case_name + '_clusters.json')
        # self.corpus = pd.read_json(path).to_dict("records")

    # Evaluate the abstract clusters
    def evaluate_abstract_clusters(self):
        try:
            # _get_small_cluster_size()
            folder = os.path.join('output', self.args.case_name + '_' + self.args.embedding_name, 'evaluation')
            path = os.path.join(folder, 'min_cluster_size_results.json')
            results = pd.read_json(path).to_dict("records")
            for result in results:
                min_cluster_size = result['min_cluster_size']
                if min_cluster_size == 2:
                    cluster_results = result['cluster_results']
                    counts = list()
                    for r in cluster_results:
                        counts.append(r['count'])
                    # Get the max count
                    max_count = np.amax(np.array(counts))
                    min_count = np.amin(np.array(counts))
                    avg_count = np.average(np.array(counts))
                    total_cluster = len(cluster_results)
                    print("Min_cluster_size " + str(min_cluster_size)
                          + " Total cluster " + str(total_cluster)
                          + " Max " + str(max_count)
                          + " Min " + str(min_count) + " Average " + str(avg_count))
        except Exception as e:
            print("Error occurred! {err}".format(err=e))

    # Evaluate the keyword group chart
    def evaluate_keyword_groups(self):
        try:
            folder = os.path.join('output', self.args.case_name + '_' + self.args.embedding_name)
            path = os.path.join(folder, self.args.case_name + '_cluster_terms_keyword_groups.json')
            clusters = pd.read_json(path).to_dict("records")
            # Collect all keyword groups' coverage
            results = list()
            # Go through each cluster
            for cluster in clusters:
                cluster_no = cluster['cluster']
                keyword_groups = cluster['keyword_groups']
                cluster_doc_ids = cluster['doc_ids']
                # Compute the coverage of each keyword group in a cluster
                for keyword_group in keyword_groups:
                    group_no = keyword_group['group']
                    keywords = keyword_group['keywords']
                    doc_ids = keyword_group['doc_ids']
                    coverage = len(doc_ids) / len(cluster_doc_ids)
                    results.append({'cluster': cluster_no,
                                    'cluster_docs': len(cluster_doc_ids),
                                    'keyword_group_no': group_no,
                                    'keywords': keywords,
                                    'score': keyword_group['score'], 'doc_ids': doc_ids,
                                    'num_docs': len(doc_ids),
                                    'num_keywords': len(keywords),
                                    'coverage': coverage
                                    })
            # # Write keyword group results to a summary (csv)
            path = os.path.join(folder, 'evaluation', 'keyword_group_evaluation.csv')
            df = pd.DataFrame(results)
            df.to_csv(path, encoding='utf-8', index=False)
        except Exception as e:
            print("Error occurred! {err}".format(err=e))

    # Evaluate topic coherence score of our keyword group
    def evaluate_topic_coherence(self):
        def compute_umass_topic_coherence_score(_cluster_docs, _topic_words):
            # Build a mapping of word and doc ids
            def _build_word_docIds(_cluster_docs, _topic_words):
                _word_docIds = {}
                for _topic_word in _topic_words:
                    try:
                        _word_docIds.setdefault(_topic_word, list())
                        # Get the number of docs containing the word
                        for _doc in _cluster_docs:
                            _doc_id = _doc['DocId']
                            _candidates = _doc['CandidatePhrases']
                            for _candidate in _candidates:
                                if _candidate['key-phrase'].lower() == _topic_word.lower():
                                    _word_docIds[_topic_word].append(_doc_id)
                                    break
                    except Exception as _err:
                        print("Error occurred! {err}".format(err=_err))
                        sys.exit(-1)
                return _word_docIds

            # # Get doc ids containing both word i and word j
            def _get_docIds_two_words(_docId_word_i, _docIds_word_j):
                return [_docId for _docId in _docId_word_i if _docId in _docIds_word_j]

            try:
                _word_docs = _build_word_docIds(_cluster_docs, _topic_words)
                total_score = 0
                _total_count = len(_topic_words)
                count = 0
                for i in range(1, _total_count):
                    try:
                        word_i = topic_words[i]
                        docs_word_i = _word_docs[word_i]
                        doc_count_word_i = len(docs_word_i)
                        for j in range(0, i - 1):
                            word_j = topic_words[j]
                            docs_word_j = _word_docs[word_j]
                            doc_word_i_j = _get_docIds_two_words(docs_word_i, docs_word_j)
                            doc_count_word_i_j = len(doc_word_i_j)
                            assert doc_count_word_i_j >= 0
                            coherence_score = math.log((doc_count_word_i_j + 1) / (1.0 * len(docs_word_j)))
                            total_score += coherence_score
                        count = count + 1
                    except Exception as _err:
                        print("Error occurred! {err}".format(err=_err))
                        sys.exit(-1)
                avg_score = 2 * total_score / (1.0 * _total_count * (_total_count - 1))
                return avg_score, _word_docs
            except Exception as _err:
                print("Error occurred! {err}".format(err=_err))

        def compute_topic_coherence_score(_topic_words, _topic_word_vectors):
            comb_counts = math.comb(len(_topic_words), 2)
            score = 0
            for i in range(0, len(topic_words) - 1):
                for j in range(i + 1, len(topic_words)):
                    try:
                        vector_i = np.array([_topic_word_vectors[i]])
                        vector_j = np.array([_topic_word_vectors[j]])
                        # Compute the similarities
                        similarities = cosine_similarity(vector_i, vector_j)
                        score += similarities[0][0]
                    except Exception as e:
                        print("Error occurred! {err}".format(err=e))
            return score / comb_counts

        try:
            # Load cluster data
            path = os.path.join('output', self.args.case_name, self.args.folder,
                                'AIMLUrbanStudyCorpus_cluster_terms_keyword_groups_updated.json')
            clusters = pd.read_json(path).to_dict("records")
            results = list()
            for cluster_no in range(1, 32):
                cluster = next(cluster for cluster in clusters if cluster['Cluster'] == cluster_no)
                keyword_groups = cluster['KeywordGroups']
                # Load keyword vectors
                path = os.path.join('output', self.args.case_name, self.args.folder,
                                    'evaluation', 'keyword_vectors',
                                    'keyword_vectors_cluster#' + str(cluster_no) + '.json')
                keyword_vectors = pd.read_json(path).to_dict("records")
                # print(keyword_vectors)
                total_count = 20
                group_no = 1
                for keyword_group in keyword_groups:
                    # Collect 50 keywords
                    topic_words = keyword_group['Key-phrases']
                    score = keyword_group['score']
                    topic_words.sort()
                    # Sort topic words by alphabetically
                    topic_words = topic_words[:total_count]
                    topic_word_vectors = list()
                    # get vectors of topic words
                    for topic_word in topic_words:
                        vector = next(
                            (v['Vectors'] for v in keyword_vectors if v['Key-phrases'].lower() == topic_word.lower()),
                            None)
                        if vector:
                            topic_word_vectors.append(vector)
                    # print(topic_words)
                    # topic_coherence_score, word_docs = compute_umass_topic_coherence_score(cluster_docs, topic_words)
                    topic_coherence_score = compute_topic_coherence_score(topic_words, topic_word_vectors)
                    print(cluster_no, "-", group_no, "-", topic_coherence_score)
                    results.append({"Cluster": cluster_no, "Group No": "Group" + str(group_no),
                                    "Topic Coherence Score": topic_coherence_score,
                                    "Silhouette Score": score})
                    group_no = group_no + 1
            # Output the clustering results of a dimension
            folder = os.path.join('output', self.args.case_name, self.args.folder, 'evaluation')
            Path(folder).mkdir(parents=True, exist_ok=True)
            # Output the detailed clustering results
            result_df = pd.DataFrame(results)
            # Output cluster results to CSV
            path = os.path.join(folder, 'topic_coherence_score_results.csv')
            result_df.to_csv(path, encoding='utf-8', index=False)
        except Exception as e:
            print("Error occurred! {err}".format(err=e))

    # Evaluate the diversity of keyword list in each abstract
    def evaluate_diversity(self):
        folder = os.path.join('output', self.args.case_name + '_' + self.args.embedding_name)
        path = os.path.join(folder, self.args.case_name + '_clusters.json')
        docs = pd.read_json(path).to_dict("records")
        doc = next(doc for doc in docs if doc['DocId'] == 477)
        # print(doc)
        candidate_words = doc['CandidatePhrases']
        abstract = doc['Abstract']
        # Set Sentence Transformer path
        sentence_transformers_path = os.path.join('/Scratch', getpass.getuser(), 'SentenceTransformer')
        # # Language model
        model = SentenceTransformer(self.args.model_name, cache_folder=sentence_transformers_path,
                                    device=self.args.device)
        # print(model)
        results = list()
        # # Rank the high scoring phrases
        for diversity in [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]:
            phrase_scores_mmr = KeywordExtractionUtility.re_rank_phrases_by_maximal_margin_relevance(
                model, abstract, candidate_words, diversity)
            mmr_keywords = list(map(lambda p: p['key-phrase'], phrase_scores_mmr))
            result = {"diversity": diversity}
            for i, mmr_keyword in enumerate(mmr_keywords[:5]):
                result[str(i) + '_keyword'] = mmr_keyword['key-phrase']
                result[str(i) + '_score'] = mmr_keyword['score']
            results.append(result)
        # Write output
        folder = os.path.join('output', self.args.case_name, self.args.folder, 'evaluation')
        Path(folder).mkdir(parents=True, exist_ok=True)
        # Output the detailed clustering results
        result_df = pd.DataFrame(results)
        # Output cluster results to CSV
        path = os.path.join(folder, 'keyword_mmr_diversity_result.csv')
        result_df.to_csv(path, encoding='utf-8', index=False)


# Main entry
if __name__ == '__main__':
    try:
        evl = Test()
        evl.evaluate_abstract_clusters()
        # evl.evaluate_keyword_groups()
        # evl.evaluate_diversity()
        # evl.evaluate_topic_coherence()
        # evl.sort_article_clusters()
        # evl.evaluate_article_clusters()
    except Exception as err:
        print("Error occurred! {err}".format(err=err))
