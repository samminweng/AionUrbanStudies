# Plots the chart to present our results in the papers
import os
from argparse import Namespace
from functools import reduce
import plotly.graph_objects as go
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd


class Evaluation:
    def __init__(self):
        self.args = Namespace(
            case_name='AIMLUrbanStudyCorpus',
            folder='cluster_merge'
        )

    # Evaluate the article cluster chart
    def evaluate_article_clusters(self):
        folder = os.path.join('output', self.args.case_name, self.args.folder)
        path = os.path.join(folder, self.args.case_name + '_cluster_terms_key_phrases_topics.json')
        clusters = pd.read_json(path).to_dict("records")
        # Sort the clusters by score
        clusters = sorted(clusters, key=lambda c: round(c['Score'], 4))
        scores = list()
        # Plot the line distribution
        for cluster in clusters:
            score = round(cluster['Score'], 2)
            if score not in scores:
                scores.append(score)
        # Sort the scores
        results = list()
        scores = sorted(scores)
        max_clusters = 0
        for score in scores:
            # Get clusters
            score_clusters = list(filter(lambda c: round(c['Score'], 2) == score, clusters))
            results.append({'score': score, 'clusters': score_clusters})
            if len(score_clusters) > max_clusters:
                max_clusters = len(score_clusters)
        # print(results)
        # Collect the counts
        fig, ax = plt.subplots()
        ax.set_xticks(np.arange(0, 1, 0.1))
        ax.set_yticks(np.arange(0, 81, 10))
        ax.set_ylim([0, 80])
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
        plt.bar(scores, group_counts[1], width=width, bottom=group_counts[0])
        plt.bar(scores, group_counts[2], width=width, bottom=group_counts[0]+group_counts[1])
        plt.bar(scores, group_counts[3], width=width, bottom=group_counts[0]+group_counts[1]+group_counts[2])
        # Add the values label
        count_values = group_counts[0]+group_counts[1]+group_counts[2]+group_counts[3]
        for ind, score in enumerate(scores):
            plt.text(score, count_values[ind] + 2, s=count_values[ind], ha='center')
        # # Add x, y axis title
        plt.xlabel('Silhouette Score', fontsize=14)
        plt.ylabel('Number of Articles', fontsize=14)

        # Save figure to file
        out_path = os.path.join(folder, 'evaluation', 'article_cluster.png')
        plt.savefig(out_path)
        plt.show()

    # Evaluate the keyword cluster chart
    def evaluate_keyword_clusters(self):
        try:
            folder = os.path.join('output', self.args.case_name, self.args.folder)
            path = os.path.join(folder, self.args.case_name + '_cluster_terms_key_phrases_topics.json')
            df = pd.read_json(path)
            # Load the corpus
            path = os.path.join(folder, self.args.case_name + '_clusters.json')
            corpus_docs = pd.read_json(path).to_dict("records")
            results = list()
            for index, row in df.iterrows():
                cluster_no = row['Cluster']
                keyword_clusters = row['KeywordClusters']
                num_topics = len(keyword_clusters)
                result = {'Cluster': cluster_no, 'NumTopics': num_topics, 'Score': 0, 'TopicWords': list()}
                top_freq_words = list()
                for group_id in range(0, num_topics):
                    if group_id < len(keyword_clusters):
                        keyword_cluster = keyword_clusters[group_id]
                        # Get score
                        score = keyword_cluster['score']
                        # Get number of key phrases
                        num_phrases = keyword_cluster['NumPhrases']
                        # Get number of doc
                        num_docs = len(keyword_cluster['DocIds'])
                        # Get topic_words
                        topic_words = keyword_cluster['TopicWords']
                        # Get DocIds
                        doc_Ids = keyword_cluster['DocIds']
                        docs = list(filter(lambda doc: doc['DocId'] in doc_Ids, corpus_docs))

                        result['keyword_cluster#' + str(group_id + 1)] = num_docs
                        result['Score'] = result['Score'] + score
                # Sort the topic word by freq
                top_freq_words = sorted(top_freq_words, key=lambda w: w['freq'], reverse=True)
                result['TopicWords'] = ', '.join(list(map(lambda w: w['word'], top_freq_words[:10])))
                # Get the average score
                result['Score'] = result['Score'] / result['NumTopics']
                results.append(result)
            # Write keyword group results to a summary (csv)
            path = os.path.join('output', self.args.case_name, self.args.folder,
                                "keyword_clusters.csv")
            df = pd.DataFrame(results)
            df.to_csv(path, encoding='utf-8', index=False)

        except Exception as err:
            print("Error occurred! {err}".format(err=err))


# Main entry
if __name__ == '__main__':
    try:
        eval = Evaluation()
        eval.evaluate_article_clusters()
    except Exception as err:
        print("Error occurred! {err}".format(err=err))