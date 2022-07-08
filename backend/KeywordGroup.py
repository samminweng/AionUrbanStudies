import os.path
import sys
from argparse import Namespace
from functools import reduce

import umap
from sentence_transformers import SentenceTransformer
from pathlib import Path
import pandas as pd
import logging
import getpass
# Set logging level
from KeywordGroupUtility import KeywordGroupUtility

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
# Set Sentence Transformer path
sentence_transformers_path = os.path.join('/Scratch', getpass.getuser(), 'SentenceTransformer')
if os.name == 'nt':
    sentence_transformers_path = os.path.join("C:", os.sep, "Users", getpass.getuser(), "SentenceTransformer")
Path(sentence_transformers_path).mkdir(parents=True, exist_ok=True)


# Extract keyword and group keywords based on the similarity
class KeywordGroup:
    def __init__(self):
        self.args = Namespace(
            case_name='AIMLUrbanStudyCorpus',
            # Model name ref: https://www.sbert.net/docs/pretrained_models.html
            model_name="all-mpnet-base-v2",
            device='cpu',
            diversity=0.5,
            cluster_folder='cluster_merge'
        )
        # Load HDBSCAN cluster
        path = os.path.join('output', self.args.case_name, self.args.cluster_folder,
                            self.args.case_name + '_clusters.json')
        self.corpus_df = pd.read_json(path)
        # Update corpus data with hdbscan cluster results
        self.corpus_df.rename(columns={'HDBSCAN_Cluster': 'Cluster'}, inplace=True)
        # Added 'Text' column
        self.corpus_df['Text'] = self.corpus_df['Title'] + ". " + self.corpus_df['Abstract']
        # Get the total cluster
        self.cluster_no_list = sorted(list(dict.fromkeys(self.corpus_df['Cluster'].tolist())))
        # self.cluster_no_list = list(range(21, 32))
        # Group all docId of a cluster
        cluster_df = self.corpus_df.groupby(['Cluster'], as_index=False).agg(
            {'DocId': lambda doc_id: list(doc_id), 'Text': lambda text: list(text)})
        cluster_df.rename(columns={'DocId': 'DocIds'}, inplace=True)
        cluster_df['NumDocs'] = cluster_df['DocIds'].apply(len)
        cluster_df = cluster_df[['Cluster', 'NumDocs', 'DocIds']]
        self.clusters = cluster_df.to_dict("records")

    # Group the key phrases with different parameters using HDBSCAN clustering
    def experiment_cluster_key_phrases(self):
        # # Language model
        model = SentenceTransformer(self.args.model_name, cache_folder=sentence_transformers_path,
                                    device=self.args.device)
        # cluster_no_list = [17]
        cluster_no_list = self.cluster_no_list
        for cluster_no in cluster_no_list:
            try:
                folder = os.path.join('output', self.args.case_name, self.args.cluster_folder, 'key_phrases')
                key_phrase_folder = os.path.join(folder, 'doc_key_phrase')
                path = os.path.join(key_phrase_folder, 'doc_key_phrases_cluster_#' + str(cluster_no) + '.json')
                df = pd.read_json(path)
                # Aggregate the key phrases of each individual paper
                all_key_phrases = reduce(lambda pre, cur: pre + cur, df['Key-phrases'].tolist(), list())
                # Filter duplicate key phrases
                key_phrases = KeywordGroupUtility.filter_unique_phrases(all_key_phrases)

                # Convert key phrases into vectors
                key_phrase_vectors = model.encode(key_phrases).tolist()
                df = pd.DataFrame()
                df['Key-phrases'] = key_phrases
                df['Vectors'] = key_phrase_vectors
                vector_folder = os.path.join(folder, 'key_phrase_vectors')
                Path(vector_folder).mkdir(parents=True, exist_ok=True)
                # Output to json or csv file
                path = os.path.join(vector_folder, 'key_phrase_vectors_cluster#' + str(cluster_no) + '.json')
                df.to_json(path, orient='records')
                # # # Cluster all key phrases using HDBSCAN clustering
                results = KeywordGroupUtility.cluster_key_phrase_experiments_by_HDBSCAN(key_phrases,
                                                                                        key_phrase_vectors,
                                                                                        is_fined_grain=False,
                                                                                        n_neighbors=len(
                                                                                            key_phrases) - 2)
                # output the experiment results
                df = pd.DataFrame(results)
                experiment_folder = os.path.join(folder, 'key_phrase_clusters', 'level_0', 'experiments')
                Path(experiment_folder).mkdir(parents=True, exist_ok=True)
                path = os.path.join(experiment_folder,
                                    'experiment_key_phrases_cluster#' + str(cluster_no) + '.json')
                df.to_json(path, orient='records')
                print("=== Complete grouping the key phrases of cluster {no} ===".format(no=cluster_no))
            except Exception as err:
                print("Error occurred! {err}".format(err=err))

    # Used the best experiment results to group the key phrases results
    def cluster_key_phrases_with_best_experiments(self):
        try:
            # Collect the best results in each cluster
            results = list()
            cluster_no_list = self.cluster_no_list
            # cluster_no_list = [17]
            for cluster_no in cluster_no_list:
                try:
                    # Output key phrases of each paper
                    key_phrase_folder = os.path.join('output', self.args.case_name, self.args.cluster_folder,
                                                     'key_phrases')
                    path = os.path.join(key_phrase_folder, 'key_phrase_clusters', 'level_0', 'experiments',
                                        'experiment_key_phrases_cluster#{c}.json'.format(c=cluster_no))
                    experiments = pd.read_json(path).to_dict("records")
                    # Sort the experiment results by score
                    experiments = sorted(experiments, key=lambda ex: (ex['score'], ex['min_cluster_size']),
                                         reverse=True)
                    # Get the best results
                    best_ex = experiments[0]
                    cluster_results = best_ex['cluster_results']
                    cluster_labels = best_ex['cluster_labels']
                    # x, y position
                    x_pos_list = best_ex['x']
                    y_pos_list = best_ex['y']
                    # Load top five key phrases of every paper in a cluster
                    path = os.path.join(key_phrase_folder, 'key_phrase_vectors',
                                        'key_phrase_vectors_cluster#{c}.json'.format(c=cluster_no))
                    vector_df = pd.read_json(path)
                    key_phrases = vector_df['Key-phrases'].tolist()
                    # Load doc key_phrase
                    path = os.path.join(key_phrase_folder, 'doc_key_phrase',
                                        'doc_key_phrases_cluster_#{c}.json'.format(c=cluster_no))
                    doc_key_phrases = pd.read_json(path).to_dict("records")
                    # # Obtain the grouped key phrases of the cluster
                    key_phrase_clusters = KeywordGroupUtility.cluster_key_phrases_with_opt_parameter(
                        key_phrases, cluster_labels, doc_key_phrases, x_pos_list, y_pos_list)
                    # # Sort the grouped key phrases by most frequent words
                    for cluster in key_phrase_clusters:
                        group_no = cluster['Group']
                        cluster_score = next(r['score'] for r in cluster_results if r['group'] == group_no)
                        cluster['Key-phrases'] = cluster['Key-phrases']
                        cluster['score'] = cluster_score
                        cluster['dimension'] = best_ex['dimension']
                        cluster['min_samples'] = best_ex['min_samples']
                        cluster['min_cluster_size'] = best_ex['min_cluster_size']
                    # Output the results to a chart
                    folder = os.path.join(key_phrase_folder, 'key_phrase_clusters', 'level_0')
                    Path(folder).mkdir(parents=True, exist_ok=True)
                    # Output the grouped key phrases
                    group_df = pd.DataFrame(key_phrase_clusters)
                    group_df = group_df[['Group', 'score', 'NumPhrases', 'Key-phrases', 'NumDocs',
                                         'DocIds', 'dimension', 'min_samples', 'min_cluster_size', 'x',
                                         'y']]  # Re-order the column list
                    path = os.path.join(folder, 'key_phrases_cluster_#' + str(cluster_no) + '.csv')
                    group_df.to_csv(path, encoding='utf-8', index=False)
                    path = os.path.join(folder, 'key_phrases_cluster_#' + str(cluster_no) + '.json')
                    group_df.to_json(path, orient='records')
                    print('Output the summary of key phrase cluster to ' + path)
                    # Store the grouped key phrases of a cluster
                    results.append({'Cluster': cluster_no, 'Key-phrases': key_phrase_clusters})
                except Exception as _err:
                    print("Error occurred! {err}".format(err=_err))
                    sys.exit(-1)
            # write best results of each group
            df = pd.DataFrame(results,
                              columns=['Cluster', 'Key-phrases'])
            folder = os.path.join('output', self.args.case_name, self.args.cluster_folder,
                                  'key_phrases', 'key_phrase_clusters')
            Path(folder).mkdir(parents=True, exist_ok=True)
            path = os.path.join(folder, 'key_phrases_cluster.csv')
            df.to_csv(path, encoding='utf-8', index=False)
            path = os.path.join(folder, 'key_phrases_cluster.json')
            df.to_json(path, orient="records")
        except Exception as err:
            print("Error occurred! {err}".format(err=err))

    # Re-cluster outlier key phrases
    def re_cluster_key_phrases_within_keyword_cluster(self):
        folder = os.path.join('output', self.args.case_name, self.args.cluster_folder, 'key_phrases')
        # Load cluster_key_phrase_groups
        path = os.path.join(folder, 'key_phrase_clusters', 'key_phrases_cluster.json')
        clusters = pd.read_json(path).to_dict("records")
        # minimal cluster size
        cluster_no_list = self.cluster_no_list
        # cluster_no_list = [17]
        try:
            for cluster_no in cluster_no_list:
                is_stop = False
                iteration = 1
                while not is_stop:  # Maximum iteration: 4
                    prev_iteration = iteration - 1
                    try:
                        cluster = next(cluster for cluster in clusters if cluster['Cluster'] == cluster_no)
                        # Load the best grouping of previous level
                        path = os.path.join(folder, 'key_phrase_clusters', 'level_' + str(prev_iteration),
                                            'key_phrases_cluster_#' + str(cluster_no) + '.json')
                        key_phrase_clusters = pd.read_json(path).to_dict("records")
                        # Load doc_key_phrases
                        path = os.path.join(folder, 'doc_key_phrase',
                                            'doc_key_phrases_cluster_#' + str(cluster_no) + ".json")
                        doc_key_phrases = pd.read_json(path).to_dict("records")
                        # Load key phrase vectors
                        path = os.path.join(folder, 'key_phrase_vectors',
                                            'key_phrase_vectors_cluster#' + str(cluster_no) + '.json')
                        key_phrase_vectors = pd.read_json(path).to_dict("records")
                        # # print(doc_key_phrases)
                        # Re-cluster keyword cluster
                        updated_key_phrase_clusters = KeywordGroupUtility.run_re_clustering_experiments(cluster_no,
                                                                                                        key_phrase_clusters,
                                                                                                        key_phrase_vectors,
                                                                                                        doc_key_phrases)

                        cluster['Key-phrases'] = updated_key_phrase_clusters
                        out_folder = os.path.join(folder, 'key_phrase_clusters', 'level_' + str(iteration))
                        Path(out_folder).mkdir(parents=True, exist_ok=True)
                        # Write new key phrase groups to
                        df = pd.DataFrame(updated_key_phrase_clusters)
                        df = df[['Group', 'score', 'NumPhrases', 'Key-phrases', 'NumDocs', 'DocIds', 'dimension',
                                 'min_samples', 'min_cluster_size', 'x', 'y']]
                        path = os.path.join(out_folder,
                                            'key_phrases_cluster_#' + str(cluster_no) + '.csv')
                        df.to_csv(path, encoding='utf-8', index=False)
                        path = os.path.join(out_folder,
                                            'key_phrases_cluster_#' + str(cluster_no) + '.json')
                        df.to_json(path, orient="records")
                        KeywordGroupUtility.visualise_keywords_cluster_results(cluster_no,
                                                                               updated_key_phrase_clusters,
                                                                               out_folder)
                        print("=== Complete re-clustering key phrases in cluster #{c_no} at Iteration {i} ===".format(
                            c_no=cluster_no, i=iteration))
                        is_stop = KeywordGroupUtility.check_stop_iteration(updated_key_phrase_clusters,
                                                                           len(doc_key_phrases))
                        iteration = iteration + 1
                    except Exception as _err:
                        print("Error occurred! {err}".format(err=_err))
                        sys.exit(-1)
            # # Write output to 'cluster_key_phrases_group'
            df = pd.DataFrame(clusters,
                              columns=['Cluster', 'Key-phrases'])
            folder = os.path.join('output', self.args.case_name, self.args.cluster_folder,
                                  'key_phrases', 'key_phrase_clusters')
            Path(folder).mkdir(parents=True, exist_ok=True)
            path = os.path.join(folder, 'key_phrases_cluster.csv')
            df.to_csv(path, encoding='utf-8', index=False)
            path = os.path.join(folder, 'key_phrases_cluster.json')
            df.to_json(path, orient="records")
        except Exception as _err:
            print("Error occurred! {err}".format(err=_err))
            sys.exit(-1)

    # Combine the TF-IDF terms and grouped key phrases results
    def combine_terms_key_phrases_results(self):
        try:
            folder = os.path.join('output', self.args.case_name, self.args.cluster_folder)
            # Combine all key phrases and TF-IDF topics to a json file
            # # Load TF-IDF topics
            path = os.path.join(folder, 'cluster_terms', self.args.case_name + '_TF-IDF_cluster_terms.json')
            topics_df = pd.read_json(path)
            cluster_df = topics_df.copy(deep=True)
            # Load grouped Key phrases
            path = os.path.join(folder, 'key_phrases', 'key_phrase_clusters', 'key_phrases_cluster.json')
            key_phrase_df = pd.read_json(path)
            cluster_df['KeyPhrases'] = key_phrase_df['Key-phrases'].tolist()
            # Re-order cluster df and Output to csv and json file
            cluster_df = cluster_df[['Cluster', 'Score', 'NumDocs', 'DocIds', 'Terms', 'FreqTerms', 'KeyPhrases']]
            folder = os.path.join(folder, 'key_phrases')
            path = os.path.join(folder, self.args.case_name + '_cluster_terms_key_phrases.csv')
            cluster_df.to_csv(path, encoding='utf-8', index=False)
            path = os.path.join(folder, self.args.case_name + '_cluster_terms_key_phrases.json')
            cluster_df.to_json(path, orient='records')
            print('Output key phrases per cluster to ' + path)
        except Exception as err:
            print("Error occurred! {err}".format(err=err))


# Main entry
if __name__ == '__main__':
    try:
        kp = KeywordGroup()
        # Extract keyword clusters
        kp.experiment_cluster_key_phrases()
        kp.cluster_key_phrases_with_best_experiments()
        kp.combine_terms_key_phrases_results()
    except Exception as err:
        print("Error occurred! {err}".format(err=err))
