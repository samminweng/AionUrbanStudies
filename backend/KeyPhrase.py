import os.path
import sys
from argparse import Namespace
from functools import reduce

from nltk import sent_tokenize, word_tokenize
from sentence_transformers import SentenceTransformer
from pathlib import Path
import pandas as pd
import logging

from BERTModelDocClusterUtility import BERTModelDocClusterUtility
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
            # case_name='CultureUrbanStudyCorpus',
            case_name='MLUrbanStudyCorpus',
            # Model name ref: https://www.sbert.net/docs/pretrained_models.html
            model_name="all-mpnet-base-v2",
            device='cpu',
            n_neighbors=3
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
    # Use RAKE score to sort the key phrases
    # Ref: https://medium.datadriveninvestor.com/rake-rapid-automatic-keyword-extraction-algorithm-f4ec17b2886c
    def extract_doc_key_phrases_by_similarity(self):
        try:
            folder = os.path.join('output', self.args.case_name, 'key_phrases', 'doc_key_phrase')
            Path(folder).mkdir(parents=True, exist_ok=True)
            corpus_docs = self.corpus_df.to_dict("records")
            # cluster_no_list = [8]
            cluster_no_list = range(-1, self.total_clusters)
            for cluster_no in cluster_no_list:
                cluster_docs = list(filter(lambda d: d['Cluster'] == cluster_no, corpus_docs))
                results = list()  # Store all the key phrases
                for doc in cluster_docs:
                    try:
                        doc_id = doc['DocId']
                        # Get the first doc
                        doc = next(doc for doc in cluster_docs if doc['DocId'] == doc_id)
                        doc_text = BERTModelDocClusterUtility.preprocess_text(doc['Text'])
                        sentences = list()
                        for sentence in sent_tokenize(doc_text):
                            tokens = word_tokenize(sentence)
                            sentences.append(tokens)
                        # Collect all the key phrases of a document
                        n_gram_candidates = []
                        for n_gram_range in [1, 2, 3]:
                            try:
                                # Extract key phrase candidates using n-gram
                                candidates = KeyPhraseUtility.generate_n_gram_candidates(sentences, n_gram_range)
                                # # find and collect top 30 key phrases similar to a paper
                                n_gram_candidates = n_gram_candidates + candidates
                            except Exception as err:
                                print("Error occurred! {err}".format(err=err))
                        candidate_scores = KeyPhraseUtility.compute_similar_score_key_phrases(self.model,
                                                                                              doc_text,
                                                                                              n_gram_candidates)

                        key_phrase_scores = KeyPhraseUtility.sort_key_phrases_by_score(candidate_scores)
                        top_key_phrases = list(map(lambda p: p['key-phrase'], key_phrase_scores))
                        # Obtain top five key phrases
                        result = {'Cluster': cluster_no, 'DocId': doc_id, 'key-phrases': top_key_phrases}
                        # Output the top 5 key-phrase and score
                        for i in range(0, 5):
                            if i < len(key_phrase_scores):
                                result['top_'+str(i)+'_phrase'] = key_phrase_scores[i]['key-phrase']
                                result['top_'+str(i)+'_score'] = key_phrase_scores[i]['score']
                            else:
                                result['top_' + str(i) + '_phrase'] = 'NAN'
                                result['top_' + str(i) + '_score'] = 0
                        results.append(result)
                    except Exception as err:
                        print("Error occurred! {err}".format(err=err))
                        sys.exit(-1)
                print(results)
                # Write key phrases to csv file
                df = pd.DataFrame(results)
                df['No'] = range(1, len(df) + 1)
                # Map the list of key phrases (dict) to a list of strings
                df = df[['No', 'Cluster', 'DocId', 'key-phrases',
                         'top_0_phrase', 'top_0_score', 'top_1_phrase', 'top_1_score',
                         'top_2_phrase', 'top_2_score', 'top_3_phrase', 'top_3_score',
                         'top_4_phrase', 'top_4_score']]  # Re-order the columns
                # Path(folder).mkdir(parents=True, exist_ok=True)
                path = os.path.join(folder, 'top_doc_key_phrases_cluster_#' + str(cluster_no) + '.csv')
                df.to_csv(path, encoding='utf-8', index=False)
                path = os.path.join(folder, 'top_doc_key_phrases_cluster_#' + str(cluster_no) + '.json')
                df.to_json(path, orient='records')
                print("Output the key phrases of cluster #" + str(cluster_no))
        except Exception as err:
            print("Error occurred! {err}".format(err=err))

    # Group the key phrases with different parameters using HDBSCAN clustering
    def group_key_phrases_by_clusters_experiments(self):
        cluster_no_list = list(range(-1, self.total_clusters))
        # cluster_no_list = [2]
        for cluster_no in cluster_no_list:
            try:
                key_phrase_folder = os.path.join('output', self.args.case_name, 'key_phrases', 'doc_key_phrase')
                path = os.path.join(key_phrase_folder, 'top_doc_key_phrases_cluster_#' + str(cluster_no) + '.json')
                df = pd.read_json(path)
                all_key_phrases = reduce(lambda pre, cur: pre + cur, df['key-phrases'].tolist(), list())
                experiment_folder = os.path.join('output', self.args.case_name, 'key_phrases', 'experiments')
                Path(experiment_folder).mkdir(parents=True, exist_ok=True)
                # # Cluster all key phrases by using HDBSCAN
                KeyPhraseUtility.group_key_phrase_experiments_by_HDBSCAN(all_key_phrases, cluster_no, self.model,
                                                                         experiment_folder, self.args.n_neighbors)
            except Exception as err:
                print("Error occurred! {err}".format(err=err))

    # Used the best experiment results to group the key phrases results
    def grouped_key_phrases_with_best_experiment_result(self):
        try:
            # Collect the best results in each cluster
            best_results = list()
            # cluster_no_list = [8]
            cluster_no_list = list(range(-1, self.total_clusters))
            for cluster_no in cluster_no_list:
                try:
                    # Output key phrases of each paper
                    key_phrase_folder = os.path.join('output', self.args.case_name, 'key_phrases')
                    path = os.path.join(key_phrase_folder, 'experiments',
                                        'top_key_phrases_cluster_#{c}_grouping_experiments.json'.format(c=cluster_no))
                    experiment_df = pd.read_json(path)
                    # Replace 'None' with None value
                    experiment_df['min_samples'] = experiment_df['min_samples'].replace('None', 0)
                    # Sort the experiment results by score
                    experiment_df = experiment_df.sort_values(['score'], ascending=False)
                    experiments = experiment_df.to_dict("records")
                    # Get the best results
                    best_result = experiments[0]
                    best_result['cluster'] = cluster_no
                    total_num_key_phrases = reduce(lambda pre, cur: pre + cur['count'], best_result['group_result'], 0)
                    best_result['total_key_phrases'] = total_num_key_phrases
                    # Load top five key phrases of every paper in a cluster
                    path = os.path.join(key_phrase_folder, 'doc_key_phrase', 'top_doc_key_phrases_cluster_#{c}.json'.format(c=cluster_no))
                    doc_key_phrases = pd.read_json(path).to_dict("records")
                    folder = os.path.join(key_phrase_folder, 'group_key_phrases', 'cluster')
                    Path(folder).mkdir(parents=True, exist_ok=True)
                    # Obtain the grouped key phrases of the cluster
                    group_key_phrases = KeyPhraseUtility.group_key_phrases_with_best_result(cluster_no,
                                                                                            best_result,
                                                                                            doc_key_phrases,
                                                                                            folder)


                    # Sort the grouped key phrases by rake
                    for group in group_key_phrases:
                        phrase_list = group['key-phrases']
                        phrase_scores = KeyPhraseUtility.compute_keyword_rake_scores(phrase_list)
                        group['key-phrases'] = list(map(lambda p: p['key-phrase'], phrase_scores))
                    best_result['grouped_key_phrases'] = group_key_phrases
                    best_results.append(best_result)
                except Exception as err:
                    print("Error occurred! {err}".format(err=err))
                    sys.exit(-1)
            # print(best_results)
            # Load best results of each group
            df = pd.DataFrame(best_results,
                              columns=['cluster', 'dimension', 'min_samples', 'min_cluster_size', 'epsilon',
                                       'total_key_phrases', 'total_groups', 'outliers', 'score', 'grouped_key_phrases'])
            folder = os.path.join('output', self.args.case_name, 'key_phrases', 'group_key_phrases')
            Path(folder).mkdir(parents=True, exist_ok=True)
            path = os.path.join(folder, 'top_key_phrases_best_grouping.csv')
            df.to_csv(path, encoding='utf-8', index=False)
            path = os.path.join(folder, 'top_key_phrases_best_grouping.json')
            df.to_json(path, orient="records")
        except Exception as err:
            print("Error occurred! {err}".format(err=err))

    # Combine the TF-IDF terms and grouped key phrases results
    def combine_terms_key_phrases_results(self):
        try:
            folder = os.path.join('output', self.args.case_name)
            # Combine all key phrases and TF-IDF topics to a json file
            # # Load TF-IDF topics
            path = os.path.join(folder, 'cluster_terms', self.args.case_name + '_TF-IDF_cluster_terms.json')
            topics_df = pd.read_json(path)
            cluster_df = topics_df.copy(deep=True)
            # Load grouped Key phrases
            path = os.path.join(folder, 'key_phrases', 'group_key_phrases', 'top_key_phrases_best_grouping.json')
            key_phrase_df = pd.read_json(path)
            cluster_df['KeyPhrases'] = key_phrase_df['grouped_key_phrases'].tolist()
            # Re-order cluster df and Output to csv and json file
            cluster_df = cluster_df[['Cluster', 'NumDocs', 'DocIds', 'Terms', 'KeyPhrases']]
            folder = os.path.join(folder, 'key_phrases')
            path = os.path.join(folder, self.args.case_name + '_cluster_terms_key_phrases.csv')
            cluster_df.to_csv(path, encoding='utf-8', index=False)
            path = os.path.join(folder, self.args.case_name + '_cluster_terms_key_phrases.json')
            cluster_df.to_json(path, orient='records')
            print('Output key phrases per cluster to ' + path)
        except Exception as err:
            print("Error occurred! {err}".format(err=err))

    # Combine clusters and doc key phrases
    def combine_cluster_doc_key_phrases(self):
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
            df = pd.DataFrame(sorted_key_phrases)
            # Combine cluster and doc key phrases
            self.corpus_df['KeyPhrases'] = df['key-phrases'].tolist()
            # Drop column
            self.corpus_df = self.corpus_df.drop('Text', axis=1)
            folder = os.path.join('output', self.args.case_name)
            path = os.path.join(folder, self.args.case_name + '_clusters.csv')
            self.corpus_df.to_csv(path, index=False, encoding='utf-8')
            path = os.path.join(folder, self.args.case_name + '_clusters.json')
            self.corpus_df.to_json(path, orient='records')
            print('Output key phrases per doc to ' + path)
        except Exception as err:
            print("Error occurred! {err}".format(err=err))


# Main entry
if __name__ == '__main__':
    try:
        kp = KeyPhraseSimilarity()
        kp.extract_doc_key_phrases_by_similarity()
        kp.group_key_phrases_by_clusters_experiments()
        kp.grouped_key_phrases_with_best_experiment_result()
        kp.combine_terms_key_phrases_results()
        kp.combine_cluster_doc_key_phrases()
    except Exception as err:
        print("Error occurred! {err}".format(err=err))
