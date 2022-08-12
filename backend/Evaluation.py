# Plots the chart to present our results in the papers
import copy
import getpass
import math
import os
import sys
from argparse import Namespace
from pathlib import Path
import plotly.graph_objects as go
import numpy as np
import pandas as pd
import seaborn as sns
import plotly.io as pio
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

from KeyWordExtractionUtility import KeywordExtractionUtility


class Evaluation:
    def __init__(self):
        self.args = Namespace(
            case_name='AIMLUrbanStudyCorpus',
            folder='cluster_merge',
            model_name="all-mpnet-base-v2",
            device='cpu',
        )
        folder = os.path.join('output', self.args.case_name, self.args.folder)
        path = os.path.join(folder, self.args.case_name + '_clusters_updated.json')
        self.corpus = pd.read_json(path).to_dict("records")

    # Sort the article clusters to make it consistent with clustered results
    def sort_article_clusters(self):
        # groups = [list(range(1, 8)), list(range(11, 18)), list(range(8, 11)), list(range(18, 32))]
        groups = [
            {"group": 1, "map": {1: 7, 2: 2, 3: 1, 4: 3, 5: 4, 6: 5, 7: 6}},
            {"group": 2, "map": {11: 8, 12: 14, 13: 12, 14: 13, 15: 11, 16: 10, 17: 9}},
            {"group": 3, "map": {8: 17, 9: 16, 10: 15}},
            {"group": 4, "map": {18: 21, 19: 19, 20: 20, 21: 18, 22: 23, 23: 24, 24: 22, 25: 26, 26: 25,
                                 27: 28, 28: 27, 29: 31, 30: 30, 31: 29}},
        ]
        try:
            folder = os.path.join('output', self.args.case_name, self.args.folder)
            path = os.path.join(folder, self.args.case_name + '_cluster_terms_keyword_groups.json')
            clusters = pd.read_json(path).to_dict("records")
            path = os.path.join(folder, self.args.case_name + '_clusters.json')
            corpus = pd.read_json(path).to_dict("records")
            # Sort the clusters by score within groups and update cluster numbers
            updated_clusters = list()
            for group in groups:
                group_no = group['group']
                group_map = group['map']
                for old_cluster_no, new_cluster_no in group_map.items():
                    update_cluster = copy.deepcopy(next(c for c in clusters if c['Cluster'] == old_cluster_no))
                    update_cluster['Group'] = group_no
                    update_cluster['Cluster'] = new_cluster_no
                    updated_clusters.append(update_cluster)
            # Sort clusters by no
            updated_clusters = sorted(updated_clusters, key=lambda c: c['Cluster'])
            # print(updated_clusters)
            updated_docs = list()
            # Update the cluster information in corpus
            for cluster in updated_clusters:
                cluster_no = cluster['Cluster']
                doc_ids = cluster['DocIds']
                docs = list(filter(lambda d: d['DocId'] in doc_ids, corpus))
                for doc in docs:
                    doc['Cluster'] = cluster_no
                    updated_docs.append(doc)
            # Write updated clusters to csv and json
            # Sorted docs by DocId
            updated_docs = sorted(updated_docs, key=lambda d: d['DocId'])
            # print(updated_docs)
            # Write clusters output
            df = pd.DataFrame(updated_clusters)
            path = os.path.join(folder, self.args.case_name + '_cluster_terms_keyword_groups_updated.csv')
            df.to_csv(path, encoding='utf-8', index=False)
            # Write article corpus to a json file
            path = os.path.join(folder,
                                self.args.case_name + '_cluster_terms_keyword_groups_updated.json')
            df.to_json(path, orient='records')
            # Write docs outputs
            df = pd.DataFrame(updated_docs)
            path = os.path.join(folder, self.args.case_name + '_clusters_updated.csv')
            df.to_csv(path, encoding='utf-8', index=False)
            # Write article corpus to a json file
            path = os.path.join(folder,
                                self.args.case_name + '_clusters_updated.json')
            df.to_json(path, orient='records')
        except Exception as e:
            print("Error occurred! {err}".format(err=e))

    #  Find the common terms
    def find_common_terms_by_clusters(self):
        # Get common terms
        def get_common_terms(_cluster_freq_terms):
            _common_terms = []
            for _freq_terms in _cluster_freq_terms:
                for _freq_term in _freq_terms:
                    _found = next((ct for ct in _common_terms if ct['term'] == _freq_term), None)
                    if _found:
                        _found['freq'] = _found['freq'] + 1
                    else:
                        _common_terms.append({'term': _freq_term, 'freq': 1})
            # Filter out common terms
            _common_terms = list(map(lambda ct: ct['term'], filter(lambda ct: ct['freq'] > 1, _common_terms)))
            return _common_terms

        # Collect the common terms from top 10 freq term
        def collect_common_terms_from_top_10_terms(_group_clusters):
            try:
                # Collect the common terms
                _common_terms = []
                _cluster_freq_terms = []
                # Get top 10 terms
                for _cluster in _group_clusters:
                    _freq_terms = list(map(lambda t: t['term'].lower(), _cluster['FreqTerms'][:10]))
                    _cluster_freq_terms.append(_freq_terms)
                # Filter out common terms
                _common_terms = get_common_terms(_cluster_freq_terms)
                _updated_cluster_terms = []
                # Filter out common terms from cluster terms
                for _cluster_term in _cluster_freq_terms:
                    _update_cluster_term = list(filter(lambda t: t not in _common_terms, _cluster_term))
                    _updated_cluster_terms.append(_update_cluster_term)
                return _common_terms, _updated_cluster_terms
            except Exception as _e:
                print("Error occurred! {err}".format(err=_e))

        try:
            folder = os.path.join('output', self.args.case_name, self.args.folder)
            path = os.path.join(folder, self.args.case_name + '_cluster_terms_keyword_groups_updated.json')
            clusters = pd.read_json(path).to_dict("records")
            updated_clusters = list()
            for group_no in range(1, 5):
                try:
                    group_clusters = list(filter(lambda c: c['Group'] == group_no, clusters))
                    common_terms, updated_cluster_terms = collect_common_terms_from_top_10_terms(group_clusters)
                    # Collect 10 ~ 20 terms
                    for r in [[10, 20], [20, 30], [30, 40], [40, 50], [50, 60], [60, 70], [70, 80]]:
                        start = r[0]
                        end = r[1]
                        index = 0
                        for updated_cluster_term, cluster in zip(updated_cluster_terms, group_clusters):
                            freq_terms = list(map(lambda t: t['term'].lower(), cluster['FreqTerms'][start:end]))
                            updated_cluster_term = updated_cluster_term + freq_terms
                            updated_cluster_terms[index] = updated_cluster_term[:10]
                            index = index + 1
                        common_terms = get_common_terms(updated_cluster_terms) + common_terms
                        # filter the cluster terms
                        for index, updated_cluster_term in enumerate(updated_cluster_terms):
                            updated_cluster_terms[index] = list(
                                filter(lambda t: t not in common_terms, updated_cluster_term))
                        # Check if each cluster has 10 term
                        is_full = True
                        for updated_cluster_term in updated_cluster_terms:
                            is_full = is_full & len(updated_cluster_term) == 10
                            # print(updated_cluster_term)
                        if is_full:
                            break
                    # Update the cluster with common terms and its frequent terms
                    for cluster, updated_cluster_term in zip(group_clusters, updated_cluster_terms):
                        cluster['CommonTerms'] = common_terms
                        cluster['ClusterTerms'] = updated_cluster_term
                        updated_clusters.append(cluster)
                    # print(updated_clusters)
                except Exception as e:
                    print("Error occurred! {err}".format(err=e))
                    sys.exit(-1)
            df = pd.DataFrame(updated_clusters, columns=['Group', 'Cluster', 'Score', 'NumDocs', 'Percent', 'DocIds',
                                                         'Terms', 'CommonTerms', 'ClusterTerms', 'FreqTerms',
                                                         'KeywordGroups'])
            path = os.path.join(folder, self.args.case_name + '_cluster_terms_keyword_groups_updated.csv')
            df.to_csv(path, encoding='utf-8', index=False)
            # Write article corpus to a json file
            path = os.path.join(folder,
                                self.args.case_name + '_cluster_terms_keyword_groups_updated.json')
            df.to_json(path, orient='records')
        except Exception as e:
            print("Error occurred! {err}".format(err=e))

    # Evaluate the article cluster chart
    def evaluate_article_clusters(self):
        # Get term per abstract cluster
        def _get_cluster_terms(_clusters, _folder):
            _results = list()
            for _cluster in _clusters:
                terms = _cluster['FreqTerms']
                _cluster_no = _cluster['Cluster']
                _result = {
                    'cluster': _cluster_no
                }
                for index, term in enumerate(terms):
                    _result['Term' + str(index)] = " " + term['term']
                _results.append(_result)
            # Write output
            _df = pd.DataFrame(_results)
            _path = os.path.join(_folder, 'evaluation', 'term_article_clusters.csv')
            _df.to_csv(_path, encoding='utf-8', index=False)

        try:
            # _get_small_cluster_size()
            folder = os.path.join('output', self.args.case_name, self.args.folder)
            path = os.path.join(folder, self.args.case_name + '_cluster_terms_keyword_groups_updated.json')
            clusters = pd.read_json(path).to_dict("records")
            _get_cluster_terms(clusters, folder)
        except Exception as e:
            print("Error occurred! {err}".format(err=e))

    # Evaluate the keyword group chart
    def evaluate_keyword_groups(self):
        # Visualise the keyword clusters
        def visualise_keywords_cluster_results(_cluster_no, _keyword_clusters,
                                               _folder):
            try:
                # filter out negative keyword clusters
                # _keyword_clusters = list(filter(lambda c: c['score'] > 0, _keyword_clusters))
                # Visualise HDBSCAN clustering results using dot chart
                colors = sns.color_palette('tab10', n_colors=len(_keyword_clusters)).as_hex()
                # Plot clustered dots and outliers
                fig = go.Figure()
                scores = list()
                x_pos = list()
                y_pos = list()
                for kp_cluster in _keyword_clusters:
                    kp_cluster_no = kp_cluster['Group']
                    score = kp_cluster['score']
                    scores.append(score)
                    marker_color = colors[kp_cluster_no - 1]
                    marker_symbol = 'circle'
                    name = 'Keyword Cluster {no}'.format(no=kp_cluster_no)
                    marker_size = 8
                    opacity = 1
                    # Add one keyword clusters
                    fig.add_trace(go.Scatter(
                        name=name,
                        mode='markers',
                        x=kp_cluster['x'],
                        y=kp_cluster['y'],
                        marker=dict(line_width=1, symbol=marker_symbol,
                                    size=marker_size, color=marker_color,
                                    opacity=opacity)
                    ))
                    x_pos = x_pos + kp_cluster['x']
                    y_pos = y_pos + kp_cluster['y']

                title = 'Article Cluster #' + str(_cluster_no)
                # Set the fixed view windows
                x_max = max(x_pos)
                x_min = min(x_pos)
                x_center = (x_max + x_min) / 2
                x_range = [min(x_center - 2, x_min - 0.5), max(x_center + 2, x_max + 0.5)]
                y_max = max(y_pos)
                y_min = min(y_pos)
                y_center = (y_max + y_min) / 2
                y_range = [min(y_center - 2, y_min - 0.5), max(y_center + 2, y_max + 0.5)]
                # Update x, y axis
                fig.update_layout(xaxis_range=x_range,
                                  yaxis_range=y_range)
                # Figure layout
                fig.update_layout(title=title,
                                  width=600, height=800,
                                  legend=dict(orientation="v"),
                                  margin=dict(l=20, r=20, t=30, b=40))

                file_name = 'keyword_cluster_#' + str(cluster_no)
                file_path = os.path.join(_folder, file_name + ".png")
                pio.write_image(fig, file_path, format='png')
                print("Output the images of clustered results to " + file_path)
                df = pd.DataFrame(_keyword_clusters)
                df = df[['Group', 'score', 'NumPhrases', 'Key-phrases', 'NumDocs', 'DocIds']]
                _path = os.path.join(_folder, file_name + '.csv')
                df.to_csv(_path, encoding='utf-8', index=False)

            except Exception as err:
                print("Error occurred! {err}".format(err=err))
                sys.exit(-1)

        try:
            folder = os.path.join('output', self.args.case_name, self.args.folder)
            path = os.path.join(folder, self.args.case_name + '_cluster_terms_keyword_groups_updated.json')
            clusters = pd.read_json(path).to_dict("records")
            # visualise_keyword_groups_by_major_cluster(clusters)
            # Collect all keyword groups
            summary = list()
            results = list()
            keyword_sizes = list()
            all_keywords = list()
            # Filter out cluster by 0.6 of score
            for cluster in clusters:
                cluster_no = cluster['Cluster']
                keyword_groups = cluster['KeywordGroups']
                cluster_doc_ids = cluster['DocIds']
                cluster_score = cluster['Score']
                img_folder = os.path.join(folder, 'evaluation', 'keyword_groups')
                Path(img_folder).mkdir(parents=True, exist_ok=True)
                # visualise_keywords_cluster_results(cluster_no, keyword_groups, img_folder)
                total_keywords = 0
                article_numbers = list()
                scores = list()
                for keyword_group in keyword_groups:
                    keywords = keyword_group['Key-phrases']
                    for keyword in keywords:
                        keyword_sizes.append(len(keyword.split(" ")))
                        all_keywords.append(keyword)
                    doc_ids = keyword_group['DocIds']
                    results.append({'ArticleCluster': cluster_no,
                                    'Article_num': len(cluster_doc_ids),
                                    'ArticleCluster_Score': cluster_score,
                                    'KeywordGroups': keyword_group['Group'],
                                    'score': keyword_group['score'],
                                    'num_keywords': len(keywords), 'Keywords': keywords,
                                    'NumDocs': len(doc_ids), 'DocIds': doc_ids
                                    })
                    article_numbers.append(len(doc_ids))
                    total_keywords += len(keywords)
                    scores.append(keyword_group['score'])
                avg_articles = np.mean(np.array(article_numbers))
                coverage = avg_articles / len(cluster_doc_ids)
                summary.append({'ArticleCluster': cluster_no,
                                'score': cluster_score,
                                'KeywordGroups': len(keyword_groups),
                                'keywords': total_keywords,
                                'coverage': coverage,
                                'Article_num': len(cluster_doc_ids),
                                'ArticlePerKeywordCluster': avg_articles})
            # # Write keyword group results to a summary (csv)
            path = os.path.join('output', self.args.case_name, self.args.folder, 'evaluation',
                                "keyword_groups.csv")
            df = pd.DataFrame(results)
            df.to_csv(path, encoding='utf-8', index=False)
            # Write the summary of keyword clusters
            path = os.path.join('output', self.args.case_name, self.args.folder, 'evaluation',
                                "keyword_groups_summary.csv")
            df = pd.DataFrame(summary)
            df.to_csv(path, encoding='utf-8', index=False)
            keyword_sizes = np.array(keyword_sizes)
            all_keywords = np.array(all_keywords)
            for s in range(6, 11):
                matches = all_keywords[keyword_sizes == s]
                print("The number of keyword of {s} size: {c}".format(s=s, c=matches.size))
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
        path = os.path.join('output', self.args.case_name, self.args.folder,
                            'AIMLUrbanStudyCorpus_clusters_updated.json')
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
        evl = Evaluation()
        evl.evaluate_diversity()
        # evl.evaluate_topic_coherence()
        # evl.sort_article_clusters()
        # evl.find_common_terms_by_clusters()
        # evl.evaluate_article_clusters()
        # evl.evaluate_keyword_groups()
    except Exception as err:
        print("Error occurred! {err}".format(err=err))
