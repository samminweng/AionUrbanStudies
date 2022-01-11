import os.path
from argparse import Namespace
from functools import reduce

from sentence_transformers import SentenceTransformer
from pathlib import Path
import pandas as pd
import logging
from KeyPhraseUtility import KeyPhraseUtility
import getpass

# Set logging level
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
# Set Sentence Transformer path
sentence_transformers_path = os.path.join('/Scratch', getpass.getuser(), 'SentenceTransformer')
if os.name == 'nt':
    sentence_transformers_path = os.path.join("C:", os.sep, "Users", getpass.getuser(), "SentenceTransformer")
Path(sentence_transformers_path).mkdir(parents=True, exist_ok=True)


# Group the key phrases based on the vector similarity
class KeyPhraseSimilarity:
    def __init__(self):
        self.args = Namespace(
            case_name='CultureUrbanStudyCorpus',
            # Model name ref: https://www.sbert.net/docs/pretrained_models.html
            model_name="all-mpnet-base-v2",
            device='cpu'
        )
        # Load HDBSCAN cluster
        path = os.path.join('output', self.args.case_name, self.args.case_name + "_clusters.json")
        self.corpus_df = pd.read_json(path)
        # Update corpus data with hdbscan cluster results
        self.corpus_df.rename(columns={'HDBSCAN_Cluster': 'Cluster'}, inplace=True)
        # Added 'Text' column
        self.corpus_df['Text'] = self.corpus_df['Title'] + ". " + self.corpus_df['Abstract']
        # Get the total cluster
        self.total_clusters = self.corpus_df['Cluster'].max() + 1
        # Language model
        self.model = SentenceTransformer(self.args.model_name, cache_folder=sentence_transformers_path,
                                         device=self.args.device)

    # # Use the BERT model to find top 5 similar key phrases of each paper
    # # Ref: https://towardsdatascience.com/keyword-extraction-with-bert-724efca412ea
    def extract_doc_key_phrases_by_clusters(self):
        try:
            corpus_docs = self.corpus_df.to_dict("records")
            cluster_no_list = range(-1, self.total_clusters)
            for cluster_no in cluster_no_list:
                cluster_docs = list(filter(lambda d: d['Cluster'] == cluster_no, corpus_docs))
                results = list()
                all_key_phrases = list()  # Store all the key phrases
                for doc in cluster_docs:
                    try:
                        doc_id = doc['DocId']
                        # Get the first doc
                        doc = next(doc for doc in cluster_docs if doc['DocId'] == doc_id)
                        sentences = KeyPhraseUtility.clean_sentence(doc['Text'])
                        doc_text = " ".join(list(map(lambda s: " ".join(s), sentences)))
                        result = {'Cluster': cluster_no, 'DocId': doc_id}
                        # Collect all the key phrases of a doc
                        candidates = []
                        for n_gram_range in [1, 2, 3]:
                            try:
                                # Extract key phrase candidates using n-gram
                                n_gram_candidates = KeyPhraseUtility.generate_n_gram_candidates(sentences,
                                                                                                n_gram_range)
                                # find and collect top 30 key phrases similar to a paper
                                top_n_gram_key_phrases = KeyPhraseUtility.get_top_similar_key_phrases(self.model,
                                                                                                      doc_text,
                                                                                                      n_gram_candidates,
                                                                                                      top_k=30)
                                result[str(n_gram_range) + '-gram-key-phrases'] = top_n_gram_key_phrases
                                candidates = candidates + list(map(lambda p: p['key-phrase'], top_n_gram_key_phrases))
                            except Exception as err:
                                print("Error occurred! {err}".format(err=err))
                        # Combine all the n-gram key phrases in a doc
                        # Get top 5 key phrase unique to all key phrase list
                        doc_key_phrases = KeyPhraseUtility.get_top_similar_key_phrases(self.model, doc_text, candidates,
                                                                                       top_k=30)
                        top_doc_key_phrases = KeyPhraseUtility.get_unique_doc_key_phrases(doc_key_phrases,
                                                                                          all_key_phrases)
                        # Write top five key phrases to 'doc_key_phrases'
                        result['key-phrases'] = top_doc_key_phrases
                        all_key_phrases = all_key_phrases + top_doc_key_phrases  # Concatenate all key phrases of a doc
                        results.append(result)
                    except Exception as err:
                        print("Error occurred! {err}".format(err=err))
                folder = os.path.join('output', self.args.case_name, 'key_phrases', 'doc_key_phrase')
                Path(folder).mkdir(parents=True, exist_ok=True)
                # Write key phrases to csv file
                KeyPhraseUtility.output_key_phrases_by_cluster(results, cluster_no, folder)
        except Exception as err:
            print("Error occurred! {err}".format(err=err))
        # Combine all the doc key phrases into a single file 'doc_key_phrases'
        try:
            key_phrase_folder = os.path.join('output', self.args.case_name, 'key_phrases')
            # Combine the key phrases of all papers to a single file
            doc_key_phrases = list()
            for cluster_no in range(-1, self.total_clusters):
                # Get key phrases of a cluster
                path = os.path.join(key_phrase_folder, 'doc_key_phrase',
                                    'top_doc_key_phrases_cluster_#{c}.json'.format(c=cluster_no))
                docs = pd.read_json(path).to_dict("records")
                for doc in docs:
                    doc_key_phrases.append({'DocId': doc['DocId'], 'key-phrases': doc['key-phrases']})
            # Sort key phrases by DocId
            sorted_key_phrases = sorted(doc_key_phrases, key=lambda k: k['DocId'])
            # # Aggregated all the key phrases of each individual article
            df = pd.DataFrame(sorted_key_phrases, columns=['DocId', 'key-phrases'])
            path = os.path.join(key_phrase_folder, self.args.case_name + '_doc_key_phrases.csv')
            df.to_csv(path, index=False, encoding='utf-8')
            path = os.path.join(key_phrase_folder, self.args.case_name + '_doc_key_phrases.json')
            df.to_json(path, orient='records')
            print('Output key phrases per doc to ' + path)
        except Exception as err:
            print("Error occurred! {err}".format(err=err))

    # Group the key phrases with different parameters using HDBSCAN clustering
    def group_key_phrases_by_clusters_experiments(self):
        # cluster_no_list = range(-1, self.total_clusters)
        cluster_no_list = [18]
        for cluster_no in cluster_no_list:
            try:
                key_phrase_folder = os.path.join('output', self.args.case_name, 'key_phrases', 'doc_key_phrase')
                path = os.path.join(key_phrase_folder, 'top_key_phrases_cluster_#' + str(cluster_no) + '.json')
                df = pd.read_json(path)
                all_key_phrases = reduce(lambda pre, cur: pre + cur, df['key-phrases'].tolist(), list())
                experiment_folder = os.path.join('output', self.args.case_name, 'key_phrases', 'experiments')
                Path(experiment_folder).mkdir(parents=True, exist_ok=True)
                # # Cluster all key phrases by using HDBSCAN
                KeyPhraseUtility.group_key_phrase_experiments_by_HDBSCAN(all_key_phrases, cluster_no, self.model,
                                                                         experiment_folder)
            except Exception as err:
                print("Error occurred! {err}".format(err=err))

    # Used the best experiment results to group the key phrases results
    def grouped_key_phrases_with_best_experiment_result(self):
        try:
            # Collect the best results in each cluster
            best_results = list()
            for cluster_no in range(-1, self.total_clusters):
                try:
                    # Output key phrases of each paper
                    folder = os.path.join('output', self.args.case_name, 'key_phrases')
                    path = os.path.join(folder, 'experiments',
                                        'top_key_phrases_cluster_#{c}_grouping_experiments.json'.format(c=cluster_no))
                    experiment_df = pd.read_json(path)
                    # Replace 'None' with None value
                    experiment_df['score'] = experiment_df['score'].replace('None', None)
                    experiment_df['min_samples'] = experiment_df['min_samples'].replace('None', 0)
                    # Sort the experiment results by score
                    experiment_df = experiment_df.sort_values(['score'], ascending=False)
                    experiments = experiment_df.to_dict("records")
                    # Get the best results
                    best_result = experiments[0]
                    best_result['cluster'] = cluster_no
                    # Load top five key phrases of every paper in a cluster
                    path = os.path.join(folder, 'doc_key_phrase',
                                        'top_doc_key_phrases_cluster_#{c}.json'.format(c=cluster_no))
                    doc_key_phrases = pd.read_json(path).to_dict("records")
                    cluster_group_key_phrase_folder = os.path.join(folder, 'group_key_phrases', 'cluster')
                    Path(cluster_group_key_phrase_folder).mkdir(parents=True, exist_ok=True)
                    # Obtain the grouped key phrases of the cluster
                    group_key_phrases = KeyPhraseUtility.group_key_phrases_with_best_result(cluster_no,
                                                                                            best_result,
                                                                                            doc_key_phrases,
                                                                                            cluster_group_key_phrase_folder)
                    best_result['grouped_key_phrases'] = group_key_phrases
                    best_results.append(best_result)
                except Exception as err:
                    print("Error occurred! {err}".format(err=err))
            # print(best_results)
            # Load best results of each cluster
            df = pd.DataFrame(best_results,
                              columns=['cluster', 'dimension', 'min_samples', 'min_cluster_size', 'epsilon',
                                       'total_groups', 'outliers', 'score', 'grouped_key_phrases'])
            folder = os.path.join('output', self.args.case_name, 'key_phrases', 'group_key_phrases')
            Path(folder).mkdir(parents=True, exist_ok=True)
            path = os.path.join(folder, 'top_key_phrases_best_grouping.csv')
            df.to_csv(path, encoding='utf-8', index=False)
            path = os.path.join(folder, 'top_key_phrases_best_grouping.json')
            df.to_json(path, orient="records")
        except Exception as err:
            print("Error occurred! {err}".format(err=err))

    # Combine the TF-IDF topics and grouped key phrases results
    def combine_topics_key_phrases_results(self):
        try:
            folder = os.path.join('output', self.args.case_name)
            # Combine all key phrases and TF-IDF topics to a json file
            # # Load TF-IDF topics
            path = os.path.join(folder, 'topics', self.args.case_name + '_TF-IDF_cluster_topics.json')
            topics_df = pd.read_json(path)
            # Load grouped Key phrases
            path = os.path.join(folder, 'key_phrases', 'group_key_phrases', 'top_key_phrases_best_grouping.json')
            key_phrase_df = pd.read_json(path)
            cluster_df = topics_df.copy(deep=True)
            cluster_df['KeyPhrases'] = key_phrase_df['grouped_key_phrases'].tolist()
            # Re-order cluster df and Output to csv and json file
            cluster_df = cluster_df[['Cluster', 'NumDocs', 'DocIds', 'Topics', 'KeyPhrases']]
            path = os.path.join(folder, self.args.case_name + '_cluster_topic_key_phrases.csv')
            cluster_df.to_csv(path, encoding='utf-8', index=False)
            path = os.path.join(folder, self.args.case_name + '_cluster_topic_key_phrases.json')
            cluster_df.to_json(path, orient='records')
            print('Output key phrases per cluster to ' + path)
        except Exception as err:
            print("Error occurred! {err}".format(err=err))


# Main entry
if __name__ == '__main__':
    kp = KeyPhraseSimilarity()
    # kp.extract_doc_key_phrases_by_clusters()
    # kp.group_key_phrases_by_clusters_experiments()
    # kp.grouped_key_phrases_with_best_experiment_result()
    kp.combine_topics_key_phrases_results()
