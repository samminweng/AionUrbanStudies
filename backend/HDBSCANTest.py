import os
from DocumentCluster import DocumentCluster
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
    # Experiment the parameters with different level
    for cluster_selection_method in ['eom', 'leaf']:
        for min_samples in range(1, 21):
            for eps in [0.0, 0.1, 0.2, 0.3, 0.4, 0.5]:
                results.append(test.doc_cluster.cluster_doc_by_hdbscan(min_cluster_size=10,
                                                                       min_samples=min_samples,
                                                                       cluster_selection_epsilon=eps,
                                                                       cluster_selection_method=cluster_selection_method))
    # Write the result to pd
    df = pd.DataFrame(results, columns=['min_cluster_size', 'min_samples', 'cluster_selection_epsilon',
                                        'cluster_selection_method', 'cluster_num', 'outliers'])
    path = os.path.join('output', 'cluster', 'experiments', 'HDBSCAN_experiment.csv')
    df.to_csv(path, encoding='utf-8', index=False)
    print("Output HDBSCAN clustering experiment results to " + path)
