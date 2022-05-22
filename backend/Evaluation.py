# Plots the chart to present our results in the papers
import os
import sys
from argparse import Namespace
from functools import reduce
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
            path = os.path.join(folder, self.args.case_name + '_cluster_terms_key_phrases_topics.json')
            clusters = pd.read_json(path).to_dict("records")
            # Sort the clusters by score
            clusters = sorted(clusters, key=lambda c: round(c['Score'], 4))
            # Collect the unique scores
            scores = np.sort(np.unique(list(map(lambda c: round(c['Score'], 4), clusters))))
            _create_freq(clusters, scores)
            _create_acc_freq(clusters, scores)
            # print(scores)
        except Exception as e:
            print("Error occurred! {err}".format(err=e))

    # Evaluate the keyword cluster chart
    def evaluate_keyword_clusters(self):
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

                avg_score = np.round(np.mean(scores), decimals=3)
                title = 'Article Cluster #' + str(_cluster_no) + ' score = ' + str(avg_score)
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
            path = os.path.join(folder, self.args.case_name + '_cluster_terms_key_phrases_topics.json')
            clusters = pd.read_json(path).to_dict("records")
            clusters = list(filter(lambda c: c['Score'] >= 0.0, clusters))
            all_keyword_clusters = list()
            results = list()
            # Filter out cluster by 0.6 of score
            for cluster in clusters:
                cluster_no = cluster['Cluster']
                keyword_clusters = cluster['KeywordClusters']
                cluster_doc_ids = cluster['DocIds']
                img_folder = os.path.join(folder, 'evaluation', 'keyword_clusters', 'positive_score')
                # img_folder = os.path.join(folder, 'evaluation', 'keyword_clusters', 'negative_score')
                visualise_keywords_cluster_results(cluster_no, keyword_clusters, img_folder)
                for keyword_cluster in keyword_clusters:
                    keywords = keyword_cluster['Key-phrases']
                    doc_ids = keyword_cluster['DocIds']
                    results.append({'ArticleCluster': cluster_no, 'Article_num': len(cluster_doc_ids),
                                    'KeywordCluster': keyword_cluster['Group'],
                                    'score': keyword_cluster['score'],
                                    'num_keywords': len(keywords), 'Keywords': keywords,
                                    'NumDocs': len(doc_ids), 'DocIds': doc_ids
                                    })
            # # Write keyword group results to a summary (csv)
            path = os.path.join('output', self.args.case_name, self.args.folder, 'evaluation',
                                "keyword_clusters.csv")
            df = pd.DataFrame(results)
            df.to_csv(path, encoding='utf-8', index=False)

            # # Get total number of keyword clusters
            # total_keyword_clusters = np.sum(list(map(lambda c: len(c['KeywordClusters']), results)))
            # print("Total number of keyword clusters = {t}".format(t=total_keyword_clusters))
            # # Get the average number of keyword clusters
            # avg_num_k_clusters = np.mean(list(map(lambda c: len(c['KeywordClusters']), results)))
            # print('Number of keyword cluster per article cluster = {c}'.format(c=avg_num_k_clusters))
            # # Get total keywords
            # total_keywords = np.sum(list(map(lambda c: c['TotalKeywords'], results)))
            # print("Total keywords = {t}".format(t=total_keywords))

            # # Load the corpus
            # path = os.path.join(folder, self.args.case_name + '_clusters.json')
            # corpus_docs = pd.read_json(path).to_dict("records")
            # results = list()
            # for index, row in df.iterrows():
            #
            #     num_topics = len(keyword_clusters)
            #
            #     top_freq_words = list()
            #     for group_id in range(0, num_topics):
            #         if group_id < len(keyword_clusters):
            #             keyword_cluster = keyword_clusters[group_id]
            #             # Get score
            #             score = keyword_cluster['score']
            #             # Get number of key phrases
            #             num_phrases = keyword_cluster['NumPhrases']
            #             # Get number of doc
            #             num_docs = len(keyword_cluster['DocIds'])
            #             # Get topic_words
            #             topic_words = keyword_cluster['TopicWords']
            #             # Get DocIds
            #             doc_Ids = keyword_cluster['DocIds']
            #             docs = list(filter(lambda doc: doc['DocId'] in doc_Ids, corpus_docs))
            #
            #             result['keyword_cluster#' + str(group_id + 1)] = num_docs
            #             result['Score'] = result['Score'] + score
            #     # Sort the topic word by freq
            #     top_freq_words = sorted(top_freq_words, key=lambda w: w['freq'], reverse=True)
            #     result['TopicWords'] = ', '.join(list(map(lambda w: w['word'], top_freq_words[:10])))
            #     # Get the average score
            #     result['Score'] = result['Score'] / result['NumTopics']
            #     results.append(result)


        except Exception as err:
            print("Error occurred! {err}".format(err=err))


# Main entry
if __name__ == '__main__':
    try:
        eval = Evaluation()
        # eval.evaluate_article_clusters()
        eval.evaluate_keyword_clusters()
    except Exception as err:
        print("Error occurred! {err}".format(err=err))
