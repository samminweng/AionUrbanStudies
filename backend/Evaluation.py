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

    # Sort the article clusters to make it consistent with clustered results
    def sort_article_clusters_by_scores(self):
        groups = [list(range(1, 8)), list(range(11, 18)), list(range(8, 11)), list(range(18, 32))]
        try:
            folder = os.path.join('output', self.args.case_name, self.args.folder)
            path = os.path.join(folder, self.args.case_name + '_cluster_terms_key_phrases_topics.json')
            clusters = pd.read_json(path).to_dict("records")
            path = os.path.join(folder, self.args.case_name + '_clusters.json')
            corpus = pd.read_json(path).to_dict("records")
            # Sort the clusters by score within groups and update cluster numbers
            current_cluster_no = 1
            updated_clusters = list()
            for group_index, group in enumerate(groups):
                grouped_clusters = copy.deepcopy(list(filter(lambda c: c['Cluster'] in group, clusters)))
                # Sort clusters by score
                grouped_clusters = sorted(grouped_clusters, key=lambda c: c['Score'], reverse=True)
                # Update the cluster no
                for grouped_cluster in grouped_clusters:
                    grouped_cluster['Group'] = group_index + 1
                    grouped_cluster['Cluster'] = current_cluster_no
                    current_cluster_no = current_cluster_no + 1
                    updated_clusters.append(grouped_cluster)
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
            path = os.path.join(folder, self.args.case_name + '_cluster_terms_key_phrases_topics_updated.csv')
            df.to_csv(path, encoding='utf-8', index=False)
            # Write article corpus to a json file
            path = os.path.join(folder,
                                self.args.case_name + '_cluster_terms_key_phrases_topics_updated.json')
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
        # Create a chart to display score and number of articles
        def _create_freq(_clusters, _scores):
            # Sort the scores
            _scores = list()
            _counts = list()
            for _cluster in _clusters:
                # Get clusters
                score = round(_cluster['Score'], 4)
                count = _cluster['NumDocs']
                _scores.append(score)
                _counts.append(count)
            results = list(zip(_scores, _counts))
            print(results)
            # Collect the counts
            fig, ax = plt.subplots()
            ax.set_xticks(np.arange(-1, 1.1, 0.2))
            # ax.set_yticks(np.arange(0, 81, 10))
            # ax.set_ylim([0, 80])
            width = 0.01
            plt.bar(np.array(scores), np.array(_counts), width=width)
            plt.xlabel('Silhouette Score', fontsize=14)
            plt.ylabel('Article Number', fontsize=14)
            # Save figure to file
            out_path = os.path.join(folder, 'evaluation', 'article_cluster.png')
            plt.savefig(out_path)
            # plt.show()

        # Draw the acc freq
        def _create_acc_freq(_clusters, _scores):
            total_numbers = reduce(lambda pre, cur: pre + len(cur['DocIds']), _clusters, 0)
            # Sort the scores
            results = list()
            for score in _scores:
                # Get clusters <= score
                score_clusters = list(filter(lambda c: round(c['Score'], 4) <= score, _clusters))
                # Get total number
                count = reduce(lambda pre, cur: pre + len(cur['DocIds']), score_clusters, 0)
                percent = int(100 * count / total_numbers)
                results.append(percent)
                # if score <= 0:
                print("Score:{s} Percent:{p} Count:{c} Cluster No:{n}".format(s=round(score, 4), p=percent, c=count,
                                                                              n=len(score_clusters)))
            fig, ax = plt.subplots()
            ax.set_xticks(np.arange(-1, 1.2, 0.2))
            ax.set_yticks(np.arange(0, 110, 10))
            plt.plot(scores, results)
            # Added a vertical line
            plt.axvline(x=0.0, color='red', ymax=1, ymin=0)
            # plt.axvline(x=0.6, color='red', ymax=1, ymin=0)
            plt.grid()
            plt.xlabel('Silhouette Score', fontsize=14)
            plt.ylabel('Accumulated Article Numbers (%)', fontsize=14)
            out_path = os.path.join(folder, 'evaluation', 'article_cluster_acc.png')
            plt.savefig(out_path)
            # plt.show()

        # Get the paramters of each article clusters
        def _get_parameters(_clusters):
            _folder = os.path.join('output', self.args.case_name)
            folder_names = ['cluster_0', 'cluster_1', 'cluster_2', 'cluster_3', 'iteration', 'cluster_-1']
            results = list()
            for folder_name in folder_names:
                iterative_folder = os.path.join(_folder, folder_name)
                # Load the updated iterative clustering summary
                _path = os.path.join(iterative_folder, 'cluster_terms', 'iterative_clusters',
                                     'AIMLUrbanStudyCorpus_iterative_summary.json')
                iterative_clusters = pd.read_json(_path).to_dict("records")
                results = results + iterative_clusters
            for _cluster in _clusters:
                doc_ids = _cluster['DocIds']
                parameters = next((result for result in results if np.array_equal(result['DocIds'], doc_ids)), None)
                assert parameters is not None, "Can not find the parameters of article clusters"
                _cluster['Dimension'] = parameters['dimension']
                _cluster['Min_Samples'] = parameters['min_samples']
                _cluster['Min_Cluster_Size'] = parameters['min_cluster_size']
                # Map the TFIDF terms
                _cluster['Terms'] = ', '.join(list(map(lambda t: t['term'], _cluster['Terms'])))
            return _clusters

        try:
            folder = os.path.join('output', self.args.case_name, self.args.folder)
            path = os.path.join(folder, self.args.case_name + '_cluster_terms_key_phrases_topics_updated.json')
            clusters = pd.read_json(path).to_dict("records")
            # Sort the clusters by score
            sorted_clusters = sorted(clusters, key=lambda c: round(c['Score'], 4))
            # Collect the unique scores
            scores = np.sort(np.unique(list(map(lambda c: round(c['Score'], 4), sorted_clusters))))
            _create_freq(sorted_clusters, scores)
            _create_acc_freq(sorted_clusters, scores)
            _get_parameters(clusters)
            df = pd.DataFrame(clusters)
            df = df[['Cluster', 'Score', 'NumDocs', 'DocIds', 'Terms', 'Dimension', 'Min_Samples', 'Min_Cluster_Size']]
            path = os.path.join(folder, 'evaluation', 'article_clusters.csv')
            df.to_csv(path, encoding='utf-8', index=False)
        except Exception as e:
            print("Error occurred! {err}".format(err=e))

    # Evaluate the keyword group chart
    def evaluate_keyword_groups(self):
        # Load keyword experiments
        def keyword_experiments():
            _folder = os.path.join('output', self.args.case_name, self.args.folder, 'key_phrases',
                                   'doc_key_phrase')
            total_scores = list()
            for _cluster_no in range(1, 32):
                _path = os.path.join(_folder, 'doc_key_phrases_cluster_#' + str(_cluster_no) + '.json')
                experiments = pd.read_json(_path).to_dict("records")
                # print(experiments)
                for ex in experiments:
                    _keywords = list(filter(lambda c: c['key-phrase'] in ex['Key-phrases'], ex['Phrase-candidates']))
                    _scores = list(map(lambda k: k['score'], _keywords))
                    total_scores = total_scores + _scores
            print("Average={avg} Min score={min} Max={max}".format(avg=np.mean(total_scores), min=np.min(total_scores),
                                                                          max=np.max(total_scores)))

        # Load keyword grouping experiments
        def keyword_group_experiments():
            _folder = os.path.join('output', self.args.case_name, self.args.folder, 'key_phrases',
                                   'key_phrase_clusters', 'level_0', 'experiments')
            for _cluster_no in range(1, 32):
                _path = os.path.join(_folder, 'experiment_key_phrases_cluster#' + str(_cluster_no) + '.json' )
                experiments = pd.read_json(_path).to_dict("records")
                # print(experiments)

        #
        def visualise_keyword_groups_by_major_cluster(_clusters):
            for group_no in range(1, 5):
                _group_clusters = list(filter(lambda c: c['Group'] == group_no, _clusters))
                # print(_group_clusters)
                for _cluster in _group_clusters:
                    keyword_groups = _cluster['KeywordClusters']
                    fig = plt.figure()
                    ax = fig.add_axes([0, 0, 1, 1])
                    _groups = list(map(lambda g: g['Group'], keyword_groups))
                    _scores = list(map(lambda g: g['score'], keyword_groups))
                    ax.bar(_groups, _scores)
                    plt.show()



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
            keyword_experiments()
            keyword_group_experiments()
            folder = os.path.join('output', self.args.case_name, self.args.folder)
            path = os.path.join(folder, self.args.case_name + '_cluster_terms_key_phrases_topics_updated.json')
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
                keyword_clusters = cluster['KeywordClusters']
                cluster_doc_ids = cluster['DocIds']
                cluster_score = cluster['Score']
                img_folder = os.path.join(folder, 'evaluation', 'keyword_clusters')
                Path(img_folder).mkdir(parents=True, exist_ok=True)
                visualise_keywords_cluster_results(cluster_no, keyword_clusters, img_folder)
                total_keywords = 0
                article_numbers = list()
                scores = list()
                for keyword_cluster in keyword_clusters:
                    keywords = keyword_cluster['Key-phrases']
                    for keyword in keywords:
                        keyword_sizes.append(len(keyword.split(" ")))
                        all_keywords.append(keyword)
                    doc_ids = keyword_cluster['DocIds']
                    results.append({'ArticleCluster': cluster_no, 'Article_num': len(cluster_doc_ids),
                                    'ArticleCluster_Score': cluster_score,
                                    'KeywordCluster': keyword_cluster['Group'],
                                    'score': keyword_cluster['score'],
                                    'num_keywords': len(keywords), 'Keywords': keywords,
                                    'NumDocs': len(doc_ids), 'DocIds': doc_ids
                                    })
                    article_numbers.append(len(doc_ids))
                    total_keywords += len(keywords)
                    scores.append(keyword_cluster['score'])
                avg_articles = np.mean(np.array(article_numbers))
                avg_score = np.mean(np.array(scores))
                coverage = avg_articles/ len(cluster_doc_ids)
                summary.append({'ArticleCluster': cluster_no,
                                'KeywordClusters': len(keyword_clusters),
                                'score': avg_score,
                                'keywords': total_keywords,
                                'coverage': coverage,
                                'Article_num': len(cluster_doc_ids),
                                'ArticlePerKeywordCluster': avg_articles})
            # # Write keyword group results to a summary (csv)
            path = os.path.join('output', self.args.case_name, self.args.folder, 'evaluation',
                                "keyword_clusters.csv")
            df = pd.DataFrame(results)
            df.to_csv(path, encoding='utf-8', index=False)
            # Write the summary of keyword clusters
            path = os.path.join('output', self.args.case_name, self.args.folder, 'evaluation',
                                "keyword_clusters_summary.csv")
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
        # evl.sort_article_clusters_by_scores()
        evl.evaluate_article_clusters()
        evl.evaluate_keyword_groups()
    except Exception as err:
        print("Error occurred! {err}".format(err=err))
