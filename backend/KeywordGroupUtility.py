import os
import sys
from functools import reduce
import hdbscan
import umap
import pandas as pd
import numpy as np
# Load function words
from nltk.corpus import stopwords
from sklearn.metrics import silhouette_score, silhouette_samples
from sklearn.metrics.pairwise import pairwise_distances
import plotly.graph_objects as go
import plotly.io as pio
import seaborn as sns


# Helper function for keyword cluster
class KeywordGroupUtility:
    stop_words = list(stopwords.words('english'))
    threshold = 40

    # Check if all small clusters have the coverage > 95%
    @staticmethod
    def check_stop_iteration(key_phrase_clusters, total_docs):
        # Get the remaining kp_clusters
        small_clusters = list(filter(lambda c: len(c['Key-phrases']) < KeywordGroupUtility.threshold, key_phrase_clusters))
        doc_ids = set()
        for r_cluster in small_clusters:
            for doc_id in r_cluster['DocIds']:
                doc_ids.add(doc_id)
        # Check if the coverage of remaining clusters > 95%
        coverage = len(doc_ids) / total_docs
        if coverage >= 0.95:
            return True

        return False

    # Filter out duplicated key phrases
    @staticmethod
    def filter_unique_phrases(key_phrases):
        unique_key_phrases = list()
        for key_phrase in key_phrases:
            found = next((k for k in unique_key_phrases if k.lower() == key_phrase.lower()), None)
            if not found:
                unique_key_phrases.append(key_phrase)
        return unique_key_phrases

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
    def cluster_key_phrases_with_opt_parameter(key_phrases, cluster_labels, doc_key_phrases, x_pos_list, y_pos_list):
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
            cluster_results = list(zip(key_phrases, cluster_labels, x_pos_list, y_pos_list))
            cluster_nos = list(set(cluster_labels))
            results = list()
            for cluster_no in cluster_nos:
                clustered_key_phrases = list(
                    map(lambda c: c[0], list(filter(lambda c: c[1] == cluster_no, cluster_results))))
                x_pos = list(
                    map(lambda c: c[2], list(filter(lambda c: c[1] == cluster_no, cluster_results))))
                y_pos = list(
                    map(lambda c: c[3], list(filter(lambda c: c[1] == cluster_no, cluster_results))))
                doc_ids = _collect_doc_ids(doc_key_phrases, clustered_key_phrases)
                results.append({
                    'Group': cluster_no,
                    'Key-phrases': clustered_key_phrases,
                    'NumPhrases': len(clustered_key_phrases),
                    # Collect doc ids containing clustered key phrases
                    'DocIds': doc_ids,
                    'NumDocs': len(doc_ids),
                    'x': x_pos,
                    'y': y_pos
                })
            return results
        except Exception as err:
            print("Error occurred! {err}".format(err=err))
            sys.exit(-1)

    # Cluster key phrases (vectors) using HDBSCAN clustering
    @staticmethod
    def cluster_key_phrase_experiments_by_HDBSCAN(key_phrases, key_phrase_vectors, is_fined_grain=False,
                                                  n_neighbors=150):
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

        dimensions = [200, 180, 150, 120, 110] + list(range(100, 5, -1))
        min_sample_list = list(range(20, 1, -1))
        min_cluster_size_list = list(range(30, 9, -1))
        # dimensions = [200, 180, 150, 120] + list(range(100, 15, -5))
        # min_sample_list = list(range(30, 1, -5))
        # min_cluster_size_list = list(range(30, 9, -1))
        # # Different parameter values
        # if is_fined_grain:
        #     dimensions = list(range(100, 5, -2))
        #     min_sample_list = list(range(20, 1, -1))

        try:
            results = list()
            # Filter out dimensions > the length of key phrases
            dimensions = list(filter(lambda d: d < len(key_phrases) - 2, dimensions))
            for dimension in dimensions:
                # Reduce the doc vectors to specific dimension
                reduced_vectors = umap.UMAP(
                    n_neighbors=n_neighbors,
                    min_dist=0.0,
                    n_components=dimension,
                    random_state=42,
                    metric="cosine").fit_transform(key_phrase_vectors)
                # Get vector dimension
                epsilon = 0.0
                # Compute the cosine distance/similarity for each doc vectors
                distances = pairwise_distances(reduced_vectors, metric='cosine')
                for min_samples in min_sample_list:
                    for min_cluster_size in min_cluster_size_list:
                        try:
                            # Group key phrase vectors using HDBSCAN clustering
                            cluster_labels = hdbscan.HDBSCAN(min_cluster_size=min_cluster_size,
                                                             min_samples=min_samples,
                                                             cluster_selection_epsilon=epsilon,
                                                             metric='precomputed').fit_predict(
                                distances.astype('float64')).tolist()
                            cluster_results = reduce(lambda pre, cur: collect_group_results(pre, cur),
                                                     cluster_labels, list())
                            if len(cluster_results) > 1:
                                cluster_vectors = distances.tolist()
                                # Compute the scores for all clustered keywords
                                # score = BERTArticleClusterUtility.compute_Silhouette_score(cluster_labels,
                                #                                                            cluster_vectors)

                                avg_score = silhouette_score(cluster_vectors, cluster_labels, metric='cosine')
                                silhouette_scores = silhouette_samples(cluster_vectors, cluster_labels, metric='cosine')
                                # Get each individual cluster's score
                                for cluster_result in cluster_results:
                                    cluster_no = cluster_result['group']
                                    cluster_silhouette_vals = silhouette_scores[np.array(cluster_labels) == cluster_no]
                                    cluster_score = np.mean(cluster_silhouette_vals)
                                    cluster_result['score'] = cluster_score
                                # print(cluster_results)
                                # Output the result
                                result = {'dimension': dimension,
                                          'min_samples': str(min_samples),
                                          'min_cluster_size': min_cluster_size,
                                          'epsilon': epsilon,
                                          'score': round(avg_score, 4),
                                          'cluster_results': cluster_results,
                                          'cluster_labels': cluster_labels,
                                          'x': list(map(lambda x: round(x, 2), reduced_vectors[:, 0])),
                                          'y': list(map(lambda y: round(y, 2), reduced_vectors[:, 1]))
                                          }
                                results.append(result)
                            # print(result)
                        except Exception as err:
                            print("Error occurred! {err}".format(err=err))
                            sys.exit(-1)
                print("[Info] Complete clustering key phrases at dimension {d}".format(d=dimension))
            # Return all experiment results
            return results
        except Exception as err:
            print("Error occurred! {err}".format(err=err))
            sys.exit(-1)

    # Run HDBSCAN experiments to re-group the phrases at 'i' iteration
    @staticmethod
    def run_re_clustering_experiments(cluster_no, key_phrase_clusters, key_phrase_vectors, doc_key_phrases):
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
            # Run the grouping experiments to regroup the key phrases
            for kp_cluster in key_phrase_clusters:
                key_phrases = kp_cluster['Key-phrases']
                vectors = list()
                for key_phrase in key_phrases:
                    vector = next(vector['Vectors'] for vector in key_phrase_vectors
                                  if vector['Key-phrases'].lower() == key_phrase.lower())
                    vectors.append(vector)
                # assert len(vectors) == len(key_phrases), "Inconsistent key phrases and vectors"
                if len(kp_cluster['Key-phrases']) < 10:
                    continue
                if len(kp_cluster['Key-phrases']) < KeywordGroupUtility.threshold:
                    results.append(kp_cluster)
                else:
                    try:
                        experiments = KeywordGroupUtility.cluster_key_phrase_experiments_by_HDBSCAN(key_phrases,
                                                                                                      vectors,
                                                                                                      is_fined_grain=True,
                                                                                                      n_neighbors=len(key_phrases) - 2)
                        # # Sort the experiments by sort
                        experiments = sorted(experiments, key=lambda ex: (ex['score'], ex['min_cluster_size']),
                                             reverse=True)
                        # Get the best experiment
                        best_ex = experiments[0]
                        dimension = best_ex['dimension']
                        min_samples = best_ex['min_samples']
                        min_cluster_size = best_ex['min_cluster_size']
                        # Get the grouping labels of key phrases
                        group_labels = np.array(best_ex['cluster_labels'])
                        group_results = best_ex['cluster_results']
                        # x, y pos
                        x_pos_arr = np.array(best_ex['x'])
                        y_pos_arr = np.array(best_ex['y'])
                        group_list = np.unique(group_labels)
                        if len(group_list) > 1:
                            np_key_phrases = np.array(key_phrases)
                            for group_no in group_list:
                                group_key_phrases = np_key_phrases[group_labels == group_no].tolist()
                                x_pos = x_pos_arr[group_labels == group_no].tolist()
                                y_pos = y_pos_arr[group_labels == group_no].tolist()
                                # print(group_key_phrases)
                                # Get Silhouette score of a group
                                score = next(r['score'] for r in group_results if r['group'] == group_no)
                                # print(sub_key_phrases)
                                doc_ids = _collect_doc_ids(doc_key_phrases, group_key_phrases)
                                results.append({'NumPhrases': len(group_key_phrases),
                                                'Key-phrases': group_key_phrases,
                                                'DocIds': doc_ids, 'NumDocs': len(doc_ids),
                                                'score': score, 'dimension': dimension, 'min_samples': min_samples,
                                                'min_cluster_size': min_cluster_size,
                                                'x': x_pos,
                                                'y': y_pos})
                        else:
                            kp_cluster['x'] = x_pos_arr.tolist()
                            kp_cluster['y'] = y_pos_arr.tolist()
                            results.append(kp_cluster)
                    except Exception as err:
                        print("Error occurred! {err}".format(err=err))
                        sys.exit(-1)
            print("[Info] Complete re-clustering the key phrases of cluster {no}".format(no=cluster_no))
            # Sort the results by number of docs
            results = sorted(results, key=lambda ex: ex['score'], reverse=True)
            # Assign group id
            group_id = 1
            for result in results:
                result['Group'] = group_id
                # Sort the key-phrases by alphabetic
                result['Key-phrases'] = sorted(result['Key-phrases'], key=str.lower)
                group_id = group_id + 1
            return results
        except Exception as err:
            print("Error occurred! {err}".format(err=err))
            sys.exit(-1)

    # Visualise the keyword clusters
    @staticmethod
    def visualise_keywords_cluster_results(cluster_no, key_phrase_clusters,
                                           folder):
        try:
            # Visualise HDBSCAN clustering results using dot chart
            colors = sns.color_palette('tab10', n_colors=len(key_phrase_clusters)).as_hex()
            # Plot clustered dots and outliers
            fig = go.Figure()
            scores = list()
            x_pos = list()
            y_pos = list()
            for kp_cluster in key_phrase_clusters:
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

            avg_score = np.round(np.mean(scores), decimals=3)
            title = 'Article Cluster #' + str(cluster_no) + ' score = ' + str(avg_score)

            if len(key_phrase_clusters) <= 4:
                x_max = round(max(x_pos) + 1.5)
                x_min = round(min(x_pos) - 1.5)
                y_max = round(max(y_pos) + 1.5)
                y_min = round(min(y_pos) - 1.5)
                # Update x, y axis
                fig.update_layout(xaxis_range=[x_min, x_max],
                                  yaxis_range=[y_min, y_max])
                # Figure layout
            fig.update_layout(title=title,
                              width=600, height=800,
                              legend=dict(orientation="v"),
                              margin=dict(l=20, r=20, t=30, b=40))

            file_name = 'key_phrases_cluster_#' + str(cluster_no)
            file_path = os.path.join(folder, file_name + ".png")
            pio.write_image(fig, file_path, format='png')
            print("Output the images of clustered results to " + file_path)
        except Exception as err:
            print("Error occurred! {err}".format(err=err))
            sys.exit(-1)
