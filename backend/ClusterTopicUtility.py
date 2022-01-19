# Helper function for cluster Similarity
from functools import reduce

import pandas as pd
from coverage.annotate import os


class ClusterTopicUtility:
    @staticmethod
    def collect_iterative_cluster_topic_results(case_name, last_iteration):
        cluster_folder = os.path.join('output', case_name, 'cluster')
        results = list()
        # Go through each iteration 1 to last iteration
        for i in range(0, last_iteration + 1):
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
                min_samples = best_result['min_samples']
                min_cluster_size = best_result['min_cluster_size']
                score = best_result['Silhouette_score']
                # Get summary of cluster topics
                folder = os.path.join(cluster_folder, 'iteration_' + str(i), 'topics')
                path = os.path.join(folder, 'TF-IDF_cluster_topic_summary.json')
                cluster_topics = pd.read_json(path).to_dict("records")
                total_papers = reduce(lambda total, ct: ct['NumDocs'] + total, cluster_topics, 0)
                for ct in cluster_topics:
                    results.append({
                        "iteration": i, "total_papers": total_papers, "dimension": dimension,
                        "min_samples": min_samples, "min_cluster_size": min_cluster_size, "score": score,
                        "Cluster": ct['Cluster'], "NumDocs": ct['NumDocs'], "Percent": ct['Percent'],
                        "DocIds": ct['DocIds']
                    })
            except Exception as _err:
                print("Error occurred! {err}".format(err=_err))
        # Load the results as data frame
        df = pd.DataFrame(results)
        # Output cluster results to CSV
        folder = os.path.join('output', case_name, 'cluster')
        path = os.path.join(folder, case_name + '_iterative_cluster_topic_summary.csv')
        df.to_csv(path, encoding='utf-8', index=False)
        path = os.path.join(folder, case_name + '_iterative_cluster_topic_summary.json')
        df.to_json(path, orient='records')
        print(df)

