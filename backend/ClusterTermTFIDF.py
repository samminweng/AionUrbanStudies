import os
import sys
from argparse import Namespace
from functools import reduce
from pathlib import Path
import pandas as pd
# Obtain the cluster results of the best results and extract cluster topics using TF-IDF
from ClusterTopicUtility import ClusterTopicUtility


class ClusterTermTFIDF:
    def __init__(self, _last_iteration, _cluster_no):
        self.args = Namespace(
            case_name='AIMLUrbanStudyCorpus',
            approach='TF-IDF',
            # cluster_folder='iteration',
            cluster_folder='cluster_' + str(_cluster_no),
            # cluster_no=_cluster_no,
            in_folder='iteration',
            last_iteration=_last_iteration
        )

    # Collect all iterative cluster results
    def collect_iterative_cluster_results(self):
        cluster_folder = os.path.join('output', self.args.case_name, self.args.cluster_folder, 'cluster')
        results = list()
        # Go through each iteration 1 to last iteration
        for i in range(0, self.args.last_iteration + 1):
            try:
                opt_dimension = 0
                # Get the best dimension
                folder = os.path.join(cluster_folder, self.args.in_folder + '_' + str(i))
                clustering_folder = os.path.join(folder, 'hdbscan_clustering')
                # Get the optimal dimension
                for file in os.listdir(clustering_folder):
                    file_name = file.lower()
                    if file_name.endswith(".png") and file_name.startswith("dimension"):
                        opt_dimension = int(file_name.split("_")[1].split(".png")[0])
                # Get the best score of all clustering experiments
                ex_folder = os.path.join(folder, 'experiments')
                path = os.path.join(ex_folder, 'HDBSCAN_cluster_doc_vector_result_summary.json')
                experiment_results = pd.read_json(path).to_dict("records")
                best_result = next(r for r in experiment_results if r['dimension'] == opt_dimension)
                min_samples = best_result['min_samples']
                min_cluster_size = best_result['min_cluster_size']
                score = best_result['Silhouette_score']
                # Get summary of cluster topics
                path = os.path.join(folder, self.args.case_name + '_cluster_docs.json')
                clusters = pd.read_json(path).to_dict("records")
                total_papers = reduce(lambda total, ct: ct['NumDocs'] + total, clusters, 0)
                for cluster in clusters:
                    results.append({
                        "iteration": i, "total_papers": total_papers, "dimension": opt_dimension,
                        "min_samples": min_samples, "min_cluster_size": min_cluster_size, "score": score,
                        "Cluster": cluster['HDBSCAN_Cluster'], "NumDocs": cluster['NumDocs'], "Percent": cluster['Percent'],
                        "DocIds": cluster['DocIds']
                    })
            except Exception as _err:
                print("Error occurred! {err}".format(err=_err))
        # Load the results as data frame
        df = pd.DataFrame(results)
        # Output cluster results to CSV
        folder = os.path.join('output', self.args.case_name, self.args.cluster_folder, 'cluster_terms', 'iterative_clusters')
        Path(folder).mkdir(parents=True, exist_ok=True)
        path = os.path.join(folder, self.args.case_name + '_iterative_summary.csv')
        df.to_csv(path, encoding='utf-8', index=False)
        path = os.path.join(folder, self.args.case_name + '_iterative_summary.json')
        df.to_json(path, orient='records')
        print(df)

    # Collect all the iterative cluster results and combine into a single cluster results
    def output_iterative_cluster_results(self):
        score_path = os.path.join('output', self.args.case_name, self.args.cluster_folder, 'cluster_terms',
                                  'iterative_clusters', self.args.case_name + '_iterative_summary.json')
        scores = pd.read_json(score_path).to_dict("records")
        # Load cluster results at 0 iteration as initial state
        cur_cluster_no = 0
        results = list()
        # Go through each iteration 1 to last iteration
        for iteration in range(0, self.args.last_iteration + 1):
            try:
                iter_score = list(filter(lambda s: s['iteration'] == iteration, scores))
                score = iter_score[0]['score']
                folder = os.path.join('output', self.args.case_name, self.args.cluster_folder, 'cluster')
                # Load the clustered docs in each iteration
                cluster_path = os.path.join(folder, self.args.in_folder + '_' + str(iteration),
                                            self.args.case_name + '_clusters.json')
                df = pd.read_json(cluster_path)
                df['Score'] = score
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
                if iteration == self.args.last_iteration:
                    results.extend(outlier_df.to_dict("records"))
                copied_results = results.copy()
                image_folder = os.path.join('output', self.args.case_name, self.args.cluster_folder,
                                            'cluster_terms', 'images')
                Path(image_folder).mkdir(parents=True, exist_ok=True)
                file_path = os.path.join(image_folder, 'iteration_' + str(iteration) + ".png")
                title = 'Iteration = ' + str(iteration)
                # Visualise the cluster results
                ClusterTopicUtility.visualise_cluster_results_by_iteration(title, copied_results, file_path)
            except Exception as _err:
                print("Error occurred! {err}".format(err=_err))
        # # Sort the results by DocID
        results = sorted(results, key=lambda c: c['DocId'])
        text_df = pd.DataFrame(results)
        # Reorder the columns
        text_df = text_df[['HDBSCAN_Cluster', 'Score', 'DocId', 'Cited by', 'Year', 'Document Type',
                            'Title', 'Abstract', 'Author Keywords', 'Authors', 'DOI', 'x', 'y']]
        # Output cluster results to CSV
        folder = os.path.join('output', self.args.case_name, self.args.cluster_folder)
        path = os.path.join(folder, self.args.case_name + '_clusters.csv')
        text_df.to_csv(path, encoding='utf-8', index=False)
        path = os.path.join(folder, self.args.case_name + '_clusters.json')
        text_df.to_json(path, orient='records')
        # print(text_df)

    # Derive the distinct from each cluster of documents
    def derive_cluster_terms_by_TF_IDF(self):
        try:
            term_folder = os.path.join('output', self.args.case_name, self.args.cluster_folder, 'cluster_terms')
            # Get the cluster docs
            path = os.path.join('output', self.args.case_name, self.args.cluster_folder,
                                self.args.case_name + '_clusters.json')
            # Load the documents clustered by
            clustered_doc_df = pd.read_json(path)
            # Update text column
            clustered_doc_df['Text'] = clustered_doc_df['Title'] + ". " + clustered_doc_df['Abstract']
            # Group the documents and doc_id by clusters
            docs_per_cluster_df = clustered_doc_df.groupby(['HDBSCAN_Cluster'], as_index=False) \
                .agg({'DocId': lambda doc_id: list(doc_id), 'Text': lambda text: list(text),
                      'Score': "mean"})
            # Get top 100 topics (1, 2, 3 grams) for each cluster
            n_gram_term_list = ClusterTopicUtility.get_n_gram_terms('HDBSCAN_Cluster',
                                                                    docs_per_cluster_df,
                                                                    term_folder)
            results = []
            for i, cluster in docs_per_cluster_df.iterrows():
                try:
                    cluster_no = cluster['HDBSCAN_Cluster']
                    score = cluster['Score']
                    doc_ids = cluster['DocId']
                    doc_texts = cluster['Text']
                    result = {"Cluster": cluster_no, "Score": score, 'NumDocs': len(doc_ids), 'DocIds': doc_ids}
                    n_gram_terms = []
                    # Collect the topics of 1 gram, 2 gram and 3 gram
                    for n_gram_range in [1, 2]:
                        n_gram_topic = next(n_gram_topic for n_gram_topic in n_gram_term_list
                                            if n_gram_topic['n_gram'] == n_gram_range)
                        # Collect top 300 topics of a cluster
                        cluster_terms = n_gram_topic['terms'][str(cluster_no)][:300]
                        # Create a mapping between the topic and its associated articles (doc)
                        doc_per_term = ClusterTopicUtility.group_docs_by_terms(n_gram_range,
                                                                               doc_ids, doc_texts,
                                                                               cluster_terms)
                        n_gram_type = 'Term-' + str(n_gram_range) + '-gram'
                        result[n_gram_type] = doc_per_term
                        n_gram_terms += doc_per_term
                    result['Term-N-gram'] = ClusterTopicUtility.merge_n_gram_terms(n_gram_terms)
                    results.append(result)
                    print('Derive term of cluster #{no}'.format(no=cluster_no))
                except Exception as _err:
                    print("Error occurred! {err}".format(err=_err))
                    sys.exit(-1)
            # Write the result to csv and json file
            cluster_df = pd.DataFrame(results, columns=['Cluster', 'Score', 'NumDocs', 'DocIds',
                                                        'Term-1-gram', 'Term-2-gram', 'Term-N-gram'])
            folder = os.path.join(term_folder, 'TF_IDF_Terms')
            Path(folder).mkdir(parents=True, exist_ok=True)
            path = os.path.join(folder, 'TF-IDF_cluster_term.csv')
            cluster_df.to_csv(path, encoding='utf-8', index=False)
            # # # Write to a json file
            path = os.path.join(folder, 'TF-IDF_cluster_term.json')
            cluster_df.to_json(path, orient='records')
            print('Output terms per cluster to ' + path)
        except Exception as err:
            print("Error occurred! {err}".format(err=err))

    #  Summarize cluster terms and output to a single file
    def summarize_cluster_terms(self):
        try:
            term_folder = os.path.join('output', self.args.case_name, self.args.cluster_folder, 'cluster_terms')
            # Load cluster topics
            path = os.path.join(term_folder, 'TF_IDF_Terms', 'TF-IDF_cluster_term.json')
            cluster_df = pd.read_json(path)
            # Write out to csv and json file
            cluster_df = cluster_df[['Cluster', 'Score', 'NumDocs', 'DocIds', 'Term-N-gram']]
            cluster_df.rename(columns={'Term-N-gram': 'Terms'}, inplace=True)
            cluster_df['Terms'] = cluster_df['Terms'].apply(lambda terms: ClusterTopicUtility.get_cluster_terms(terms, 10))
            # total_clusters = cluster_df['Cluster'].max() + 1
            # # # Output top 50 topics by 1, 2 and 3-grams at specific cluster
            # for cluster_no in range(-1, total_clusters):
            #     folder = os.path.join(term_folder, 'n_grams')
            #     ClusterTopicUtility.flatten_tf_idf_terms(cluster_no, folder)
            # # Output cluster df to csv or json file
            path = os.path.join(term_folder, self.args.case_name + '_TF-IDF_cluster_terms.csv')
            cluster_df.to_csv(path, encoding='utf-8', index=False)
            path = os.path.join(term_folder, self.args.case_name + '_TF-IDF_cluster_terms.json')
            cluster_df.to_json(path, orient='records')
            # Output a summary of top 10 Topics of each cluster
            clusters = cluster_df.to_dict("records")
            summary_df = cluster_df.copy(deep=True)
            total = summary_df['NumDocs'].sum()
            summary_df['Percent'] = list(map(lambda c: c['NumDocs'] / total, clusters))
            summary_df['Terms'] = list(
                map(lambda c: ", ".join(list(map(lambda t: t['term'], c['Terms'][:10]))), clusters))
            summary_df = summary_df.reindex(columns=['Cluster', 'Score', 'NumDocs', 'Percent', 'DocIds', 'Terms'])
            # Output the summary as csv
            path = os.path.join(term_folder, self.args.case_name + '_TF-IDF_cluster_terms_summary.csv')
            summary_df.to_csv(path, encoding='utf-8', index=False)
        except Exception as err:
            print("Error occurred! {err}".format(err=err))


# Main entry
if __name__ == '__main__':
    try:
        cluster_no = 3
        last_iteration = 0
        ct = ClusterTermTFIDF(last_iteration, cluster_no)
        ct.collect_iterative_cluster_results()
        ct.output_iterative_cluster_results()
        ct.derive_cluster_terms_by_TF_IDF()
        ct.summarize_cluster_terms()
    except Exception as err:
        print("Error occurred! {err}".format(err=err))
