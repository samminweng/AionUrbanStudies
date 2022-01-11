import os
from argparse import Namespace
from pathlib import Path

import pandas as pd

# Extract Cluster Topic using TF-IDF
from BERTModelDocClusterUtility import BERTModelDocClusterUtility


class ClusterTopic:
    def __init__(self, _last_iteration):
        self.args = Namespace(
            case_name='CultureUrbanStudyCorpus',
            approach='TF-IDF',
            last_iteration=_last_iteration
        )

    # Collect all the iterative cluster results and combine into a single cluster results
    def collect_iterative_cluster_results(self):
        folder = os.path.join('output', self.args.case_name, 'cluster')
        cur_cluster_no = 0
        results = list()
        for i in list(range(0, self.args.last_iteration + 1)):
            try:
                cluster_path = os.path.join(folder, 'iteration_' + str(i), self.args.case_name + '_clusters.json')
                df = pd.read_json(cluster_path)
                cluster_df = df[df['HDBSCAN_Cluster'] != -1]
                total_cluster_no = cluster_df['HDBSCAN_Cluster'].max() + 1
                # Added the clustered results
                for cluster_no in range(0, total_cluster_no):
                    c_df = cluster_df[cluster_df['HDBSCAN_Cluster'] == cluster_no]
                    c_df.loc[:, 'HDBSCAN_Cluster'] = cur_cluster_no + cluster_no
                    # Updated the cluster no
                    c_results = c_df.to_dict("records")
                    results.extend(c_results)
                # Added the outliers if it reaches the last iteration
                if i == self.args.last_iteration:
                    outlier_df = df[df['HDBSCAN_Cluster'] == -1]
                    results.extend(outlier_df.to_dict("records"))
                cur_cluster_no = cur_cluster_no + total_cluster_no
            except Exception as _err:
                print("Error occurred! {err}".format(err=_err))
        # print(results)
        # Sort the results by DocID
        results = sorted(results, key=lambda c: c['DocId'])
        text_df = pd.DataFrame(results, columns=['HDBSCAN_Cluster', 'DocId', 'Cited by', 'Year', 'Document Type',
                                                 'Title', 'Abstract', 'Author Keywords', 'Authors', 'DOI', 'x',
                                                 'y'])
        # Output cluster results to CSV
        path = os.path.join(folder, self.args.case_name + '_clusters.csv')
        text_df.to_csv(path, encoding='utf-8', index=False)
        path = os.path.join(folder, self.args.case_name + '_clusters.json')
        text_df.to_json(path, orient='records')
        print(text_df)

    # Derive the topic from each cluster of documents
    def derive_cluster_topics_by_TF_IDF(self):
        # approach = 'HDBSCAN_Cluster'
        try:
            topic_folder = os.path.join('output', self.args.case_name, 'topics')
            Path(topic_folder).mkdir(parents=True, exist_ok=True)
            path = os.path.join('output', self.args.case_name, 'cluster', self.args.case_name + '_clusters.json')
            # Load the documents clustered by
            clustered_doc_df = pd.read_json(path)
            # Update text column
            clustered_doc_df['Text'] = clustered_doc_df['Title'] + ". " + clustered_doc_df['Abstract']
            # Group the documents and doc_id by clusters
            docs_per_cluster_df = clustered_doc_df.groupby(['HDBSCAN_Cluster'], as_index=False) \
                .agg({'DocId': lambda doc_id: list(doc_id), 'Text': lambda text: list(text)})
            # Get top 100 topics (1, 2, 3 grams) for each cluster
            n_gram_topic_list = BERTModelDocClusterUtility.get_n_gram_topics('HDBSCAN_Cluster',
                                                                             docs_per_cluster_df,
                                                                             topic_folder, is_load=False)
            results = []
            for i, cluster in docs_per_cluster_df.iterrows():
                try:
                    cluster_no = cluster['HDBSCAN_Cluster']
                    doc_ids = cluster['DocId']
                    doc_texts = cluster['Text']
                    result = {"Cluster": cluster_no, 'NumDocs': len(doc_ids), 'DocIds': doc_ids}
                    n_gram_topics = []
                    # Collect the topics of 1 gram, 2 gram and 3 gram
                    for n_gram_range in [1, 2, 3]:
                        n_gram_topic = next(n_gram_topic for n_gram_topic in n_gram_topic_list
                                            if n_gram_topic['n_gram'] == n_gram_range)
                        # Collect top 300 topics of a cluster
                        cluster_topics = n_gram_topic['topics'][str(cluster_no)][:300]
                        # Create a mapping between the topic and its associated articles (doc)
                        doc_per_topic = BERTModelDocClusterUtility.group_docs_by_topics(n_gram_range,
                                                                                        doc_ids, doc_texts,
                                                                                        cluster_topics)
                        n_gram_type = 'Topic-' + str(n_gram_range) + '-gram'
                        result[n_gram_type] = doc_per_topic
                        n_gram_topics += doc_per_topic
                    result['Topic-N-gram'] = BERTModelDocClusterUtility.merge_n_gram_topic(n_gram_topics)
                    results.append(result)
                    print('Derive topics of cluster #{no}'.format(no=cluster_no))
                except Exception as _err:
                    print("Error occurred! {err}".format(err=_err))
            # Write the result to csv and json file
            cluster_df = pd.DataFrame(results, columns=['Cluster', 'NumDocs', 'DocIds',
                                                        'Topic-1-gram', 'Topic-2-gram', 'Topic-3-gram',
                                                        'Topic-N-gram'])
            folder = os.path.join(topic_folder, 'n_grams')
            Path(folder).mkdir(parents=True, exist_ok=True)
            path = os.path.join(folder, 'TF-IDF_cluster_topic_n_grams.csv')
            cluster_df.to_csv(path, encoding='utf-8', index=False)
            # # # Write to a json file
            path = os.path.join(folder, 'TF-IDF_cluster_topic_n_grams.json')
            cluster_df.to_json(path, orient='records')
            print('Output topics per cluster to ' + path)
        except Exception as err:
            print("Error occurred! {err}".format(err=err))

    #  Summarize cluster topics and output cluster topics to a single file
    def summarize_cluster_topics(self):
        try:
            topic_folder = os.path.join('output', self.args.case_name, 'topics')
            # Load cluster topics
            path = os.path.join(topic_folder, 'n_grams', 'TF-IDF_cluster_topic_n_grams.json')
            cluster_df = pd.read_json(path)
            # Write out to csv and json file
            cluster_df = cluster_df.reindex(columns=['Cluster', 'NumDocs', 'DocIds', 'Topic-N-gram'])
            cluster_df.rename(columns={'Topic-N-gram': 'Topics'}, inplace=True)
            total_clusters = cluster_df['Cluster'].max() + 1
            # # Output top 50 topics by 1, 2 and 3-grams at specific cluster
            for cluster_no in range(-1, total_clusters):
                folder = os.path.join(topic_folder, 'n_grams')
                BERTModelDocClusterUtility.flatten_tf_idf_topics(cluster_no, folder)
            # # Output cluster df to csv or json file
            path = os.path.join(topic_folder, self.args.case_name + '_TF-IDF_cluster_topics.csv')
            cluster_df.to_csv(path, encoding='utf-8', index=False)
            path = os.path.join(topic_folder, self.args.case_name + '_TF-IDF_cluster_topics.json')
            cluster_df.to_json(path, orient='records')
            # Output a summary of top 10 Topics of each cluster
            clusters = cluster_df.to_dict("records")
            summary_df = cluster_df.copy(deep=True)
            total = summary_df['NumDocs'].sum()
            summary_df['Percent'] = list(map(lambda c: c['NumDocs'] / total, clusters))
            summary_df['Topics'] = list(
                map(lambda c: ", ".join(list(map(lambda t: t['topic'], c['Topics'][:10]))), clusters))
            summary_df = summary_df.reindex(columns=['Cluster', 'NumDocs', 'Percent', 'DocIds', 'Topics'])
            # Output the summary as csv
            path = os.path.join(topic_folder, self.args.case_name + '_TF-IDF_cluster_topic_summary.csv')
            summary_df.to_csv(path, encoding='utf-8', index=False)
        except Exception as err:
            print("Error occurred! {err}".format(err=err))


# Main entry
if __name__ == '__main__':
    try:
        last_iteration = 6
        ct = ClusterTopic(6)
        # ct.collect_iterative_cluster_results()
        # ct.derive_cluster_topics_by_TF_IDF()
        ct.summarize_cluster_topics()
    except Exception as err:
        print("Error occurred! {err}".format(err=err))
