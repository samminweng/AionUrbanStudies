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
            n_neighbors=3,
            diversity=0.0
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
    def extract_doc_key_phrases_by_similarity_diversity(self):
        try:
            folder = os.path.join('output', self.args.case_name, 'key_phrases', 'doc_key_phrase')
            Path(folder).mkdir(parents=True, exist_ok=True)
            corpus_docs = self.corpus_df.to_dict("records")
            # cluster_no_list = [0]
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
                            except Exception as __err:
                                print("Error occurred! {err}".format(err=__err))
                        candidate_scores = KeyPhraseUtility.compute_similar_score_key_phrases(self.model,
                                                                                              doc_text,
                                                                                              n_gram_candidates)

                        phrase_similar_scores = KeyPhraseUtility.sort_phrases_by_similar_score(candidate_scores)
                        phrase_candidates = list(map(lambda p: p['key-phrase'], phrase_similar_scores))
                        # Rank  top 20 high scoring phrases
                        num = 20
                        diversity = 0.5
                        phrase_scores_mmr = KeyPhraseUtility.re_rank_phrases_by_maximal_margin_relevance(
                            self.model, doc_text, phrase_candidates[:num], diversity)
                        key_phrases = list(map(lambda p: p['key-phrase'], phrase_scores_mmr))
                        # Obtain top five key phrases
                        result = {'Cluster': cluster_no, 'DocId': doc_id,  # 'top_num': num, 'Diversity': diversity,
                                  'Key-phrases': key_phrases[:5], 'Candidate-count': len(phrase_candidates),
                                  'Phrase-candidates': phrase_candidates}
                        # Output the top 5 key-phrase and score
                        # for i in range(0, 20):
                        #     if i < len(phrase_scores_mmr):
                        #         result['top_'+str(i)+'_phrase'] = phrase_scores_mmr[i]['key-phrase']
                        #         result['top_'+str(i)+'_score'] = phrase_scores_mmr[i]['score']
                        #     else:
                        #         result['top_' + str(i) + '_phrase'] = 'NAN'
                        #         result['top_' + str(i) + '_score'] = 0
                        results.append(result)
                        print("Complete to extract the key phrases from document {d_id}".format(d_id=doc_id))
                    except Exception as _err:
                        print("Error occurred! {err}".format(err=_err))
                        sys.exit(-1)
                print(results)
                # Write key phrases to csv file
                df = pd.DataFrame(results)
                # Map the list of key phrases (dict) to a list of strings
                Path(folder).mkdir(parents=True, exist_ok=True)
                path = os.path.join(folder, 'doc_key_phrases_cluster_#' + str(cluster_no) + '.csv')
                df.to_csv(path, encoding='utf-8', index=False)
                path = os.path.join(folder, 'doc_key_phrases_cluster_#' + str(cluster_no) + '.json')
                df.to_json(path, orient='records')
                print("Output the key phrases of cluster #" + str(cluster_no))
        except Exception as err:
            print("Error occurred! {err}".format(err=err))

    # Group the key phrases with different parameters using HDBSCAN clustering
    def experiment_group_cluster_key_phrases(self):
        # cluster_no_list = list(range(-1, self.total_clusters))
        cluster_no_list = [0]
        for cluster_no in cluster_no_list:
            try:
                key_phrase_folder = os.path.join('output', self.args.case_name, 'key_phrases', 'doc_key_phrase')
                path = os.path.join(key_phrase_folder, 'doc_key_phrases_cluster_#' + str(cluster_no) + '.json')
                df = pd.read_json(path)
                # Aggregate the key phrases of each individual paper
                all_key_phrases = reduce(lambda pre, cur: pre + cur, df['Key-phrases'].tolist(), list())
                # Filter duplicate key phrases
                unique_key_phrases = list()
                for key_phrase in all_key_phrases:
                    found = next((k for k in unique_key_phrases if k.lower() == key_phrase.lower()), None)
                    if not found:
                        unique_key_phrases.append(key_phrase)
                experiment_folder = os.path.join('output', self.args.case_name, 'key_phrases', 'experiments', 'level_0')
                Path(experiment_folder).mkdir(parents=True, exist_ok=True)
                min_cluster_size_list = list(range(30, 14, -1))
                # # Cluster all key phrases using HDBSCAN clustering
                results = KeyPhraseUtility.group_key_phrase_experiments_by_HDBSCAN(unique_key_phrases,
                                                                                   min_cluster_size_list,
                                                                                   self.model,
                                                                                   self.args.n_neighbors)
                # Update the 'parent_group'
                for result in results:
                    result['parent_group'] = 'root'
                # output the experiment results
                df = pd.DataFrame(results)
                path = os.path.join(experiment_folder,
                                    'key_phrases_cluster_#' + str(cluster_no) + '_grouping_experiments.csv')
                df.to_csv(path, encoding='utf-8', index=False)
                path = os.path.join(experiment_folder,
                                    'key_phrases_cluster_#' + str(cluster_no) + '_grouping_experiments.json')
                df.to_json(path, orient='records')
                print("=== Complete grouping the key phrases of cluster {no} ===".format(no=cluster_no))
            except Exception as err:
                print("Error occurred! {err}".format(err=err))

    # Used the best experiment results to group the key phrases results
    def group_cluster_key_phrases_with_best_experiments(self):
        try:
            # Collect the best results in each cluster
            cluster_no_list = [0]
            # cluster_no_list = list(range(-1, self.total_clusters))
            for cluster_no in cluster_no_list:
                try:
                    # Output key phrases of each paper
                    key_phrase_folder = os.path.join('output', self.args.case_name, 'key_phrases')
                    path = os.path.join(key_phrase_folder, 'experiments', 'level_0',
                                        'key_phrases_cluster_#{c}_grouping_experiments.json'.format(c=cluster_no))
                    experiments = pd.read_json(path).to_dict("records")
                    # Sort the experiment results by score
                    experiments = sorted(experiments, key=lambda ex: (ex['score'], ex['dimension']), reverse=True)
                    # Get the best results
                    optimal_parameter = experiments[0]
                    optimal_parameter['Cluster'] = cluster_no
                    optimal_parameter['total_key_phrases'] = reduce(lambda pre, cur: pre + cur['count'],
                                                                    optimal_parameter['group_results'], 0)
                    # Load top five key phrases of every paper in a cluster
                    path = os.path.join(key_phrase_folder, 'doc_key_phrase',
                                        'doc_key_phrases_cluster_#{c}.json'.format(c=cluster_no))
                    doc_key_phrases = pd.read_json(path).to_dict("records")
                    # Obtain the grouped key phrases of the cluster
                    group_key_phrases = KeyPhraseUtility.group_cluster_key_phrases_with_opt_parameter(optimal_parameter,
                                                                                                      doc_key_phrases)
                    # Sort the grouped key phrases by rake
                    for group in group_key_phrases:
                        phrase_scores = KeyPhraseUtility.rank_key_phrases_by_rake_scores(group['Key-phrases'])
                        group['Key-phrases'] = list(map(lambda p: p['key-phrase'], phrase_scores))
                        group['score'] = optimal_parameter['score']
                        group['dimension'] = optimal_parameter['dimension']
                        group['min_samples'] = optimal_parameter['min_samples']
                        group['min_cluster_size'] = optimal_parameter['min_cluster_size']
                    # Output the grouped key phrases
                    group_df = pd.DataFrame(group_key_phrases)
                    group_df['Cluster'] = cluster_no
                    group_df['Parent'] = 'root'
                    group_df = group_df[['Cluster', 'Parent', 'Group', 'NumPhrases', 'Key-phrases', 'NumDocs',
                                         'DocIds', 'score', 'dimension', 'min_samples',
                                         'min_cluster_size']]  # Re-order the column list
                    folder = os.path.join(key_phrase_folder, 'group_key_phrases', 'level_0')
                    Path(folder).mkdir(parents=True, exist_ok=True)
                    path = os.path.join(folder, 'group_key_phrases_cluster_#' + str(cluster_no) + '.csv')
                    group_df.to_csv(path, encoding='utf-8', index=False)
                    path = os.path.join(folder, 'group_key_phrases_cluster_#' + str(cluster_no) + '.json')
                    group_df.to_json(path, orient='records')
                    print('Output the summary of grouped key phrase to ' + path)
                except Exception as err:
                    print("Error occurred! {err}".format(err=err))
                    sys.exit(-1)
            # optimal_parameter['grouped_key_phrases'] = group_key_phrases
            # best_results.append(optimal_parameter)
            # # print(best_results)
            # # Load best results of each group
            # df = pd.DataFrame(best_results,
            #                   columns=['Cluster', 'dimension', 'min_samples', 'min_cluster_size', 'epsilon',
            #                            'total_key_phrases', 'total_groups', 'outliers', 'score', 'grouped_key_phrases'])
            # folder = os.path.join('output', self.args.case_name, 'key_phrases', 'group_key_phrases')
            # Path(folder).mkdir(parents=True, exist_ok=True)
            # path = os.path.join(folder, 'cluster_key_phrases_best_grouping.csv')
            # df.to_csv(path, encoding='utf-8', index=False)
            # path = os.path.join(folder, 'cluster_key_phrases_best_grouping.json')
            # df.to_json(path, orient="records")
        except Exception as err:
            print("Error occurred! {err}".format(err=err))

    # Re-group the key phrases within a group
    def re_group_key_phrases_within_groups(self):
        key_phrase_folder = os.path.join('output', self.args.case_name, 'key_phrases')
        # cluster_no_list = list(range(-1, self.total_clusters))
        min_cluster_size_list = list(range(30, 9, -1))
        level = 4
        cluster_no_list = [0]
        try:
            for cluster_no in cluster_no_list:
                KeyPhraseUtility.run_re_grouping_experiments(level, cluster_no, key_phrase_folder,
                                                             min_cluster_size_list,
                                                             self.model, self.args.n_neighbors)
                # # # Re-group the key phrases within a group using the optimal parameters
                KeyPhraseUtility.re_group_phrases_by_opt_experiment(level, cluster_no, key_phrase_folder)
        except Exception as _err:
            print("Error occurred! {err}".format(err=_err))
            sys.exit(-1)

    # Aggregate all the sub-groups to level 1
    def flat_re_grouped_key_phrases(self):
        cluster_no = 0
        last_level = 4
        all_sub_groups = list()
        # Collect all the groups and sub-groups
        for level in range(2, last_level + 1):
            # Load parent level
            folder = os.path.join('output', self.args.case_name, 'key_phrases', 'group_key_phrases',
                                  'level_' + str(level))
            path = os.path.join(folder, 'group_key_phrases_cluster_#' + str(cluster_no) + '.json')
            # Get the large groups
            all_sub_groups = all_sub_groups + pd.read_json(path).to_dict("records")
        # Load the groups at level 1
        folder = os.path.join('output', self.args.case_name, 'key_phrases', 'group_key_phrases',
                              'level_1')
        path = os.path.join(folder, 'group_key_phrases_cluster_#' + str(cluster_no) + '.json')
        # Get the large groups
        results = list()
        groups = pd.read_json(path).to_dict("records")
        for group in groups:
            parent = group['Parent']
            group_id = group['Group']
            group['TitleWords'] = KeyPhraseUtility.get_top_frequent_words(group['Key-phrases'])
            if len(group['Key-phrases']) <= 30:
                results.append(group)
            else:
                # Find all the sub-groups starting with 'root_1'
                sub_groups = list(filter(lambda g: g['Parent'].startswith(parent + '_' + str(group_id)) and
                                                   len(g['Key-phrases']) <= 30, all_sub_groups))

                # Update the parents of sub-group
                for sub_group in sub_groups:
                    sub_group['Parent'] = parent
                    sub_group['TitleWords'] = KeyPhraseUtility.get_top_frequent_words(sub_group['Key-phrases'])
                results = results + sub_groups
        # Get all the unique parent ids
        parent_ids = list(set(map(lambda g: g['Parent'], results)))
        # Sort parent ids
        parent_ids = sorted(parent_ids, key=lambda p: int(p.split("_")[1]))
        sorted_results = list()
        for parent_id in parent_ids:
            sub_groups = list(filter(lambda g: g['Parent'] == parent_id, results))
            # Sort the sub groups by num phrases
            sub_groups = sorted(sub_groups, key=lambda g: int(g['NumPhrases']), reverse=True)
            # Re-number the sub-group
            g_id = 0
            for sub_group in sub_groups:
                sub_group['Group'] = g_id
                g_id += 1
            sorted_results = sorted_results + sub_groups
        # # print(best_results)
        # # Load best results of each group
        df = pd.DataFrame(sorted_results, columns=['Cluster', 'Parent', 'Group', 'TitleWords', 'NumPhrases', 'Key-phrases', 'NumDocs', 'DocIds'])
        df = df.rename(columns={'Parent': 'Group', 'Group': 'SubGroup'})
        folder = os.path.join('output', self.args.case_name, 'key_phrases', 'group_key_phrases')
        Path(folder).mkdir(parents=True, exist_ok=True)
        path = os.path.join(folder, 'cluster_key_phrases_sub_grouping_cluster_#' + str(cluster_no) + '.csv')
        df.to_csv(path, encoding='utf-8', index=False)
        path = os.path.join(folder, 'cluster_key_phrases_sub_grouping_cluster_#' + str(cluster_no) + '.json')
        df.to_json(path, orient="records")
        print(results)
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
        # kp.extract_doc_key_phrases_by_similarity_diversity()
        # kp.experiment_group_cluster_key_phrases()
        # kp.group_cluster_key_phrases_with_best_experiments()
        # kp.re_group_key_phrases_within_groups()
        kp.flat_re_grouped_key_phrases()
        # kp.combine_terms_key_phrases_results()
        # kp.combine_cluster_doc_key_phrases()
    except Exception as err:
        print("Error occurred! {err}".format(err=err))
