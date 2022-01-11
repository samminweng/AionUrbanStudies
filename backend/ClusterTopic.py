import os
from argparse import Namespace

# Extract Cluster Topic using TF-IDF
import pandas as pd


class ClusterTopic:
    def __init__(self, _last_iteration):
        self.args = Namespace(
            case_name='CultureUrbanStudyCorpus',
            approach='TF-IDF',
            last_iteration=_last_iteration
        )
        self.text_df = None

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
        self.text_df = pd.DataFrame(results, columns=['HDBSCAN_Cluster', 'DocId', 'Cited by', 'Year', 'Document Type',
                                                      'Title', 'Abstract', 'Author Keywords', 'Authors', 'DOI', 'x',
                                                      'y'])
        # Output cluster results to CSV
        path = os.path.join(folder, self.args.case_name + '_clusters.csv')
        self.text_df.to_csv(path, encoding='utf-8', index=False)
        path = os.path.join(folder, self.args.case_name + '_clusters.json')
        self.text_df.to_json(path, orient='records')
        print(self.text_df)


# Main entry
if __name__ == '__main__':
    try:
        last_iteration = 6
        ct = ClusterTopic(6)
        ct.collect_iterative_cluster_results()
    except Exception as err:
        print("Error occurred! {err}".format(err=err))
