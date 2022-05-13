# Plots the chart to present our results in the papers
import os
from argparse import Namespace

import pandas as pd


class Evaluation:
    def __init__(self):
        self.args = Namespace(
            case_name='AIMLUrbanStudyCorpus',
            folder='cluster_merge'
        )

    # Collect and generate statistics from results
    def collect_topic_statistics(self):
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
