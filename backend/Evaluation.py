# Plots the chart to present our results in the papers
import copy
import os
import sys
from argparse import Namespace
from functools import reduce
from pathlib import Path

import plotly.graph_objects as go
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import plotly.io as pio
import plotly.figure_factory as ff


class Evaluation:
    def __init__(self):
        self.args = Namespace(
            case_name='AIMLUrbanStudyCorpus',
            folder='cluster_merge'
        )
        folder = os.path.join('output', self.args.case_name, self.args.folder)
        path = os.path.join(folder, self.args.case_name + '_clusters_updated.json')
        self.corpus = pd.read_json(path).to_dict("records")

    # Sort the article clusters to make it consistent with clustered results
    def sort_article_clusters_by_scores(self):
        # groups = [list(range(1, 8)), list(range(11, 18)), list(range(8, 11)), list(range(18, 32))]
        groups = [
            {"group": 1, "map": {1:7, 2: 2, 3: 1, 4:3, 5: 4, 6: 5, 7:6}},
            {"group": 2, "map": {11:8, 12:14, 13:12, 14:13, 15:11, 16:10, 17: 9}},
            {"group": 3, "map": {8:17, 9: 16, 10: 15}},
            {"group": 4, "map": {18: 21, 19: 19, 20: 20, 21:18, 22:23, 23:24, 24:22, 25:26, 26:25,
                                 27:28, 28:27, 29:31, 30:30, 31:29}},
             ]
        try:
            folder = os.path.join('output', self.args.case_name, self.args.folder)
            path = os.path.join(folder, self.args.case_name + '_cluster_terms_keyword_groups_updated.json')
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
            print(updated_clusters)
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

    # Evaluate the article cluster chart
    def evaluate_article_clusters(self):
        # Get term per abstract cluster
        def _get_cluster_terms(_clusters, _folder):
            _results = list()
            for _cluster in _clusters:
                terms = _cluster['Terms']
                _cluster_no = _cluster['Cluster']
                _result = {
                    'cluster': _cluster_no
                }
                for index, term in enumerate(terms):
                    _result['Term' + str(index)] = term['term']
                    # _result['Freq' + str(index)] = term['freq']
                    # _result['Range' + str(index)] = len(term['cluster_ids'])
                    # _result['Score' + str(index)] = term['score']
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
            # _get_parameters(clusters)
            # df = pd.DataFrame(clusters)
            # df = df[['Cluster', 'Score', 'NumDocs', 'DocIds', 'Terms', 'Dimension', 'Min_Samples', 'Min_Cluster_Size']]
            # path = os.path.join(folder, 'evaluation', 'article_clusters.csv')
            # df.to_csv(path, encoding='utf-8', index=False)
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


# Main entry
if __name__ == '__main__':
    try:
        evl = Evaluation()
        evl.sort_article_clusters_by_scores()
        # evl.evaluate_article_clusters()
        # evl.evaluate_keyword_groups()
    except Exception as err:
        print("Error occurred! {err}".format(err=err))
