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


class Evaluation:
    def __init__(self):
        self.args = Namespace(
            case_name='AIMLUrbanStudyCorpus',
            folder='cluster_merge'
        )

    # Sort the article clusters to make it consistent with clustered results
    def sort_article_clusters_by_scores(self):
        groups = [list(range(1, 8)), list(range(8, 11)), list(range(11, 18)), list(range(18, 21)),
                  list(range(21, 23)), range(23, 26), list(range(27, 32))]
        try:
            folder = os.path.join('output', self.args.case_name, self.args.folder)
            path = os.path.join(folder, self.args.case_name + '_cluster_terms_key_phrases_topics_updated.json')
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
            results = list()
            max_clusters = 0
            for score in _scores:
                # Get clusters
                score_clusters = list(filter(lambda c: round(c['Score'], 4) == score, _clusters))
                results.append({'score': score, 'clusters': score_clusters})
                if len(score_clusters) > max_clusters:
                    max_clusters = len(score_clusters)
            # print(results)
            # Collect the counts
            fig, ax = plt.subplots()
            ax.set_xticks(np.arange(-1, 1.2, 0.2))
            # ax.set_yticks(np.arange(0, 81, 10))
            # ax.set_ylim([0, 80])
            width = 0.01
            group_counts = list()
            for ind in range(0, max_clusters):
                counts = list()
                for result in results:
                    r_clusters = result['clusters']
                    if ind < len(r_clusters):
                        counts.append(r_clusters[ind]['NumDocs'])
                    else:
                        counts.append(0)
                group_counts.append(np.array(counts))
            plt.bar(scores, group_counts[0], width=width)
            plt.xlabel('Silhouette Score', fontsize=14)
            plt.ylabel('Article Number', fontsize=14)
            # Save figure to file
            out_path = os.path.join(folder, 'evaluation', 'article_cluster.png')
            plt.savefig(out_path)
            plt.show()

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
            plt.show()

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
            # print(scores)
            # Map the TFIDF terms
            terms = list(map(lambda c: ', '.join(list(map(lambda t: t['term'], c['Terms']))), clusters))
            df = pd.DataFrame(clusters)
            df = df[['Cluster', 'Score', 'NumDocs', 'DocIds']]
            df['Terms'] = terms
            path = os.path.join(folder, 'evaluation', 'article_clusters.csv')
            df.to_csv(path, encoding='utf-8', index=False)

        except Exception as e:
            print("Error occurred! {err}".format(err=e))

    # Evaluate the keyword cluster chart
    def evaluate_keyword_clusters(self):
        # Visualise the keyword clusters
        def visualise_keywords_cluster_results(_cluster_no, _keyword_clusters,
                                               _folder, _weight_avg_score):
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

                title = 'Article Cluster #' + str(_cluster_no) + ' score = ' + str(_weight_avg_score)
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

        def get_weight_average(_keyword_clusters):
            _total_weight = 0
            _total_sum = 0
            for _keyword_cluster in _keyword_clusters:
                _weight = len(_keyword_cluster['Key-phrases'])
                _sum = _weight * _keyword_cluster['score']
                _total_weight += _weight
                _total_sum += _sum
            return round(_total_sum/_total_weight, 2)

        try:
            folder = os.path.join('output', self.args.case_name, self.args.folder)
            path = os.path.join(folder, self.args.case_name + '_cluster_terms_key_phrases_topics_updated.json')
            clusters = pd.read_json(path).to_dict("records")
            # Filter cluster by score
            # clusters = list(filter(lambda c: c['Score'] >= 0.0, clusters))
            results = list()
            # Filter out cluster by 0.6 of score
            for cluster in clusters:
                cluster_no = cluster['Cluster']
                keyword_clusters = cluster['KeywordClusters']
                cluster_doc_ids = cluster['DocIds']
                cluster_score = cluster['Score']
                weight_average = get_weight_average(keyword_clusters)
                img_folder = os.path.join(folder, 'evaluation', 'keyword_clusters')
                Path(img_folder).mkdir(parents=True, exist_ok=True)
                visualise_keywords_cluster_results(cluster_no, keyword_clusters, img_folder, weight_average)
                for keyword_cluster in keyword_clusters:
                    keywords = keyword_cluster['Key-phrases']
                    doc_ids = keyword_cluster['DocIds']
                    results.append({'ArticleCluster': cluster_no, 'Article_num': len(cluster_doc_ids),
                                    'ArticleCluster_Score': cluster_score,
                                    'KeywordCluster': keyword_cluster['Group'],
                                    'score': keyword_cluster['score'],
                                    'num_keywords': len(keywords), 'Keywords': keywords,
                                    'NumDocs': len(doc_ids), 'DocIds': doc_ids,
                                    'weight_average': weight_average
                                    })
            # # Write keyword group results to a summary (csv)
            path = os.path.join('output', self.args.case_name, self.args.folder, 'evaluation',
                                "keyword_clusters.csv")
            df = pd.DataFrame(results)
            df.to_csv(path, encoding='utf-8', index=False)
        except Exception as e:
            print("Error occurred! {err}".format(err=e))


# Main entry
if __name__ == '__main__':
    try:
        evl = Evaluation()
        # evl.re_number_article_clusters_by_scores()
        evl.evaluate_article_clusters()
        # evl.evaluate_keyword_clusters()
    except Exception as err:
        print("Error occurred! {err}".format(err=err))
