import os
from documentCluster import DocumentCluster
import pandas as pd


# Test HDBSCAN cluster
class HDBSCANTest:
    def __init__(self):
        self.doc_cluster = DocumentCluster()


# Main entry
if __name__ == '__main__':
    # Experiment HDBSCAN parameters (min_cluster_size, min_samples, cluster_selection_epsilon)
    test = HDBSCANTest()
    results = []
    # Experiment the parameters with different levels
    # for min_cluster_size in range(2, 21):
    for cluster_selection_method in ['eom', 'leaf']:
        for min_samples in range(1, 21):
            results.append(test.doc_cluster.cluster_doc_by_hdbscan(min_cluster_size=10,
                                                                   min_samples=min_samples,
                                                                   cluster_selection_epsilon=0.0,
                                                                   cluster_selection_method=cluster_selection_method))
    # Write the result to pd
    df = pd.DataFrame(results, columns=['min_cluster_size', 'min_samples', 'cluster_selection_method',
                                        'cluster_num', 'outliers'])
    path = os.path.join('output', 'cluster', 'experiments', 'HDBSCAN_experiment.csv')
    df.to_csv(path, encoding='utf-8', index=False)
    print("Output HDBSCAN clustering experiment results to " + path)
