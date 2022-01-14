import os
from argparse import Namespace
from functools import reduce
from pathlib import Path

import pandas as pd

from BERTModelDocClusterUtility import BERTModelDocClusterUtility

# Obtain the cluster results of the best results and extract cluster topics using TF-IDF
class ClusterTopic:
    def __init__(self, _last_iteration):
        self.args = Namespace(
            case_name='CultureUrbanStudyCorpus',
            approach='TF-IDF',
            last_iteration=_last_iteration
        )

    def collect_iterative_cluster_topic_results(self):
        cluster_folder = os.path.join('output', self.args.case_name, 'cluster')
        results = list()
        # Go through each iteration 1 to last iteration
        for i in range(0, self.args.last_iteration + 1):
            try:
                dimension = 0
                # Get the best dimension
                folder = os.path.join(cluster_folder, 'iteration_' + str(i), 'hdbscan_clustering')
                for file in os.listdir(folder):
                    file_name = file.lower()
                    if file_name.endswith(".png") and file_name.startswith("dimension"):
                        dimension = int(file_name.split("_")[1].split(".png")[0])
                # Get the best score
                folder = os.path.join(cluster_folder, 'iteration_' + str(i), 'experiments')
                path = os.path.join(folder, 'HDBSCAN_cluster_doc_vector_result_summary.json')
                experiment_results = pd.read_json(path).to_dict("records")
                best_result = next(r for r in experiment_results if r['dimension'] == dimension)
                score = best_result['Silhouette_score']
                # Get summary of cluster topics
                folder = os.path.join(cluster_folder, 'iteration_' + str(i), 'topics')
                path = os.path.join(folder, 'TF-IDF_cluster_topic_summary.json')
                df = pd.read_json(path)
                cluster_topics = df.to_dict("records")
                total_papers = reduce(lambda ct1, total: ct1['NumDocs'] + total, cluster_topics, 0)
                for ct in cluster_topics:
                    results.append({
                        "iteration": i, "total_papers": total_papers, "dimension": dimension, "score": score,
                        "cluster": ct['Cluster'], "NumDocs": ct['NumDocs'], "Percent": ct['Percent'],
                        "DocIds": ct['DocIds'], "Topics": ct['Topics']
                    })
            except Exception as _err:
                print("Error occurred! {err}".format(err=_err))
        print(results)


    # Collect all the iterative cluster results and combine into a single cluster results
    def collect_iterative_cluster_results(self):
        folder = os.path.join('output', self.args.case_name, 'cluster')
        # Load cluster results at 0 iteration as initial state
        cur_cluster_no = 0
        results = list()
        # Go through each iteration 1 to last iteration
        for i in range(0, self.args.last_iteration + 1):
            try:
                cluster_path = os.path.join(folder, 'iteration_' + str(i), self.args.case_name + '_clusters.json')
                df = pd.read_json(cluster_path)
                cluster_df = df[df['HDBSCAN_Cluster'] != -1]
                total_cluster_no = cluster_df['HDBSCAN_Cluster'].max()
                cluster_no_list = list(range(0, total_cluster_no + 1))
                # Added the clustered results
                for cluster_no in cluster_no_list:
                    # Get the clustered docs
                    c_df = cluster_df[cluster_df['HDBSCAN_Cluster'] == cluster_no]
                    docs = c_df.to_dict("records")
                    for doc in docs:
                        doc['HDBSCAN_Cluster'] = cur_cluster_no + cluster_no
                    results.extend(docs)
                cur_cluster_no = cur_cluster_no + len(cluster_no_list)
                # Get outliers
                outlier_df = df[df['HDBSCAN_Cluster'] == -1]
                # visual_results.extend(outlier_df.to_dict("records"))
                # Add the outliers at lst iteration
                if i == self.args.last_iteration:
                    results.extend(outlier_df.to_dict("records"))
                copied_results = results.copy()
                image_folder = os.path.join('output', self.args.case_name, 'topics', 'images')
                Path(image_folder).mkdir(parents=True, exist_ok=True)
                # Visualise the cluster results
                BERTModelDocClusterUtility.visualise_cluster_results_by_iteration(i, copied_results, image_folder)
            except Exception as _err:
                print("Error occurred! {err}".format(err=_err))
        # # Sort the results by DocID
        results = sorted(results, key=lambda c: c['DocId'])
        text_df = pd.DataFrame(results, columns=['HDBSCAN_Cluster', 'DocId', 'Cited by', 'Year', 'Document Type',
                                                 'Title', 'Abstract', 'Author Keywords', 'Authors', 'DOI', 'x',
                                                 'y'])
        # Output cluster results to CSV
        folder = os.path.join('output', self.args.case_name)
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
            path = os.path.join('output', self.args.case_name, self.args.case_name + '_clusters.json')
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
        last_iteration = 10
        ct = ClusterTopic(last_iteration)
        ct.collect_iterative_cluster_topic_results()
        # ct.collect_iterative_cluster_results()
        # ct.derive_cluster_topics_by_TF_IDF()
        # ct.summarize_cluster_topics()
    except Exception as err:
        print("Error occurred! {err}".format(err=err))
