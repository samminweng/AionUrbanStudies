import os.path
import sys
from argparse import Namespace
from functools import reduce
from nltk import sent_tokenize, word_tokenize
from sentence_transformers import SentenceTransformer
from pathlib import Path
import pandas as pd
import logging

from BERTArticleClusterUtility import BERTArticleClusterUtility
import getpass

# Set logging level
from KeywordClusterUtility import KeywordClusterUtility

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
# Set Sentence Transformer path
sentence_transformers_path = os.path.join('/Scratch', getpass.getuser(), 'SentenceTransformer')
if os.name == 'nt':
    sentence_transformers_path = os.path.join("C:", os.sep, "Users", getpass.getuser(), "SentenceTransformer")
Path(sentence_transformers_path).mkdir(parents=True, exist_ok=True)


# Extract keyword and group keywords based on the similarity
class KeywordCluster:
    # def __init__(self, _cluster_no):
    def __init__(self):
        self.args = Namespace(
            case_name='AIMLUrbanStudyCorpus',
            # Model name ref: https://www.sbert.net/docs/pretrained_models.html
            model_name="all-mpnet-base-v2",
            device='cpu',
            diversity=0.5,
            cluster_folder='cluster_merge',
            # cluster_folder='cluster_' + str(_cluster_no),
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
        # self.cluster_no_list = [3, 8]
        # Group all docId of a cluster
        cluster_df = self.corpus_df.groupby(['Cluster'], as_index=False).agg(
            {'DocId': lambda doc_id: list(doc_id), 'Text': lambda text: list(text)})
        cluster_df.rename(columns={'DocId': 'DocIds'}, inplace=True)
        cluster_df['NumDocs'] = cluster_df['DocIds'].apply(len)
        cluster_df = cluster_df[['Cluster', 'NumDocs', 'DocIds']]
        self.clusters = cluster_df.to_dict("records")
        # # Language model
        self.model = SentenceTransformer(self.args.model_name, cache_folder=sentence_transformers_path,
                                         device=self.args.device)

    # # Use the BERT model to find top 5 similar key phrases of each paper
    # # Ref: https://towardsdatascience.com/keyword-extraction-with-bert-724efca412ea
    def extract_doc_key_phrases_by_similarity_diversity(self):
        try:
            folder = os.path.join('output', self.args.case_name, self.args.cluster_folder,
                                  'key_phrases', 'doc_key_phrase')
            Path(folder).mkdir(parents=True, exist_ok=True)
            corpus_docs = self.corpus_df.to_dict("records")
            # Collect all the tfidf terms from all docs
            # # A folder that stores all the topic results
            tfidf_folder = os.path.join(folder, 'tf-idf')
            Path(tfidf_folder).mkdir(parents=True, exist_ok=True)
            # Extract single-word candidates using TF-IDF
            tfidf_candidates = KeywordClusterUtility.generate_tfidf_terms(corpus_docs, tfidf_folder)
            # Collect collocation from each cluster of articles
            # cluster_no_list = [8]
            cluster_no_list = self.cluster_no_list
            for cluster_no in cluster_no_list:
                cluster_docs = list(filter(lambda d: d['Cluster'] == cluster_no, corpus_docs))
                results = list()  # Store all the key phrases
                for doc in cluster_docs:
                    doc_id = doc['DocId']
                    # if doc_id != 376:
                    #     continue
                    # Get the first doc
                    doc = next(doc for doc in cluster_docs if doc['DocId'] == doc_id)
                    doc_text = BERTArticleClusterUtility.preprocess_text(doc['Abstract'])
                    sentences = list()
                    for sentence in sent_tokenize(doc_text):
                        tokens = word_tokenize(sentence)
                        sentences.append(tokens)
                    # End of for loop
                    try:
                        doc_tfidf_candidates = next(c for c in tfidf_candidates if c['doc_id'] == doc_id)['terms']
                        # Get top 2 uni_grams from tf-idf terms
                        uni_gram_candidates = doc_tfidf_candidates[:2]
                        # Collect all the candidate collocation words
                        n_gram_candidates = KeywordClusterUtility.generate_collocation_candidates(sentences)
                        n_gram_candidates = n_gram_candidates + list(map(lambda c: c['term'], uni_gram_candidates))
                        # print(", ".join(n_gram_candidates))
                        candidate_scores = KeywordClusterUtility.compute_similar_score_key_phrases(self.model,
                                                                                                   doc_text,
                                                                                                   n_gram_candidates)

                        phrase_similar_scores = KeywordClusterUtility.sort_phrases_by_similar_score(candidate_scores)
                        phrase_candidates = list(map(lambda p: p['key-phrase'], phrase_similar_scores))
                        # Rank the high scoring phrases
                        phrase_scores_mmr = KeywordClusterUtility.re_rank_phrases_by_maximal_margin_relevance(
                            self.model, doc_text, phrase_candidates, self.args.diversity)
                        mmr_key_phrases = list(map(lambda p: p['key-phrase'], phrase_scores_mmr))
                        # filter out single word overlapping with any other
                        top_key_phrases = list()
                        for key_phrase in mmr_key_phrases:
                            if len(key_phrase.split(" ")) == 1:
                                single_word = key_phrase.lower()
                                # Check if the single word overlaps with existing words
                                found = next((phrase for phrase in top_key_phrases if single_word != phrase.lower() and
                                              single_word in phrase.lower()), None)
                                if not found:
                                    top_key_phrases.append(key_phrase)
                            else:
                                top_key_phrases.append(key_phrase)

                        # Obtain top five key phrases
                        result = {'Cluster': cluster_no, 'DocId': doc_id,
                                  'Key-phrases': top_key_phrases[:5],
                                  'Candidate-count': len(phrase_similar_scores),
                                  'Phrase-candidates': phrase_similar_scores}
                        results.append(result)
                        print("Complete to extract the key phrases from document {d_id}".format(d_id=doc_id))
                    except Exception as __err:
                        print("Error occurred! {err}".format(err=__err))
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

    # Combine clusters and doc key phrases
    def combine_cluster_doc_key_phrases(self):
        # Combine all the doc key phrases into a single file 'doc_key_phrases'
        try:
            key_phrase_folder = os.path.join('output', self.args.case_name, self.args.cluster_folder, 'key_phrases')
            # Combine the key phrases of all papers to a single file
            doc_key_phrases = list()
            for cluster_no in self.cluster_no_list:
                # Get key phrases of a cluster
                path = os.path.join(key_phrase_folder, 'doc_key_phrase',
                                    'doc_key_phrases_cluster_#{c}.json'.format(c=cluster_no))
                docs = pd.read_json(path).to_dict("records")
                for doc in docs:
                    doc_key_phrases.append({'DocId': doc['DocId'], 'KeyPhrases': doc['Key-phrases']})
            # Sort key phrases by DocId
            sorted_key_phrases = sorted(doc_key_phrases, key=lambda k: k['DocId'])
            # # Aggregated all the key phrases of each individual article
            df = pd.DataFrame(sorted_key_phrases)
            # Combine cluster and doc key phrases
            self.corpus_df['KeyPhrases'] = df['KeyPhrases'].tolist()
            # Drop column
            self.corpus_df = self.corpus_df.drop('Text', axis=1)
            folder = os.path.join('output', self.args.case_name, self.args.cluster_folder)
            path = os.path.join(folder, self.args.case_name + '_clusters.csv')
            self.corpus_df.to_csv(path, index=False, encoding='utf-8')
            path = os.path.join(folder, self.args.case_name + '_clusters.json')
            self.corpus_df.to_json(path, orient='records')
            print('Output key phrases per doc to ' + path)
        except Exception as err:
            print("Error occurred! {err}".format(err=err))

    # Group the key phrases with different parameters using HDBSCAN clustering
    def experiment_group_cluster_key_phrases(self):
        cluster_no_list = self.cluster_no_list
        # cluster_no_list = [8]
        for cluster_no in cluster_no_list:
            try:
                key_phrase_folder = os.path.join('output', self.args.case_name, self.args.cluster_folder,
                                                 'key_phrases', 'doc_key_phrase')
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
                experiment_folder = os.path.join('output', self.args.case_name, self.args.cluster_folder,
                                                 'key_phrases', 'experiments', 'level_0')
                Path(experiment_folder).mkdir(parents=True, exist_ok=True)

                # # Cluster all key phrases using HDBSCAN clustering
                results = KeywordClusterUtility.group_key_phrase_experiments_by_HDBSCAN(unique_key_phrases,
                                                                                        self.model)
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
            results = list()
            cluster_no_list = self.cluster_no_list
            # cluster_no_list = [8]
            for cluster_no in cluster_no_list:
                try:
                    # Output key phrases of each paper
                    key_phrase_folder = os.path.join('output', self.args.case_name, self.args.cluster_folder,
                                                     'key_phrases')
                    path = os.path.join(key_phrase_folder, 'experiments', 'level_0',
                                        'key_phrases_cluster_#{c}_grouping_experiments.json'.format(c=cluster_no))
                    experiments = pd.read_json(path).to_dict("records")
                    # Sort the experiment results by score
                    experiments = sorted(experiments, key=lambda ex: (ex['score'], ex['min_cluster_size']),
                                         reverse=True)
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
                    group_key_phrases = KeywordClusterUtility.group_cluster_key_phrases_with_opt_parameter(
                        optimal_parameter,
                        doc_key_phrases)
                    # Sort the grouped key phrases by most frequent words
                    for group in group_key_phrases:
                        group['Key-phrases'] = group['Key-phrases']
                        group['score'] = optimal_parameter['score']
                        group['dimension'] = optimal_parameter['dimension']
                        group['min_samples'] = optimal_parameter['min_samples']
                        group['min_cluster_size'] = optimal_parameter['min_cluster_size']
                    # Store the grouped key phrases of a cluster
                    results.append({'Cluster': cluster_no, 'Key-phrases': group_key_phrases})
                    # Output the grouped key phrases
                    group_df = pd.DataFrame(group_key_phrases)
                    # group_df['Cluster'] = cluster_no
                    group_df['Parent'] = 'root'
                    group_df = group_df[['Group', 'NumPhrases', 'Key-phrases', 'NumDocs',
                                         'DocIds', 'score', 'dimension', 'min_samples',
                                         'min_cluster_size']]  # Re-order the column list
                    folder = os.path.join(key_phrase_folder, 'group_key_phrases', 'keyword_cluster')
                    Path(folder).mkdir(parents=True, exist_ok=True)
                    path = os.path.join(folder, 'group_key_phrases_cluster_#' + str(cluster_no) + '.csv')
                    group_df.to_csv(path, encoding='utf-8', index=False)
                    path = os.path.join(folder, 'group_key_phrases_cluster_#' + str(cluster_no) + '.json')
                    group_df.to_json(path, orient='records')
                    print('Output the summary of grouped key phrase to ' + path)
                except Exception as _err:
                    print("Error occurred! {err}".format(err=_err))
                    sys.exit(-1)
            # write best results of each group
            df = pd.DataFrame(results,
                              columns=['Cluster', 'Key-phrases'])
            folder = os.path.join('output', self.args.case_name, self.args.cluster_folder,
                                  'key_phrases', 'group_key_phrases')
            Path(folder).mkdir(parents=True, exist_ok=True)
            path = os.path.join(folder, 'cluster_key_phrases_group.csv')
            df.to_csv(path, encoding='utf-8', index=False)
            path = os.path.join(folder, 'cluster_key_phrases_group.json')
            df.to_json(path, orient="records")
        except Exception as err:
            print("Error occurred! {err}".format(err=err))

    # Re-group the key phrases within a group
    def re_group_key_phrases_within_keyword_cluster(self):
        folder = os.path.join('output', self.args.case_name, self.args.cluster_folder, 'key_phrases')
        # Load cluster_key_phrase_groups
        path = os.path.join(folder, 'group_key_phrases', 'cluster_key_phrases_group.json')
        clusters = pd.read_json(path).to_dict("records")
        # minimal cluster size
        cluster_no_list = self.cluster_no_list
        try:
            for cluster_no in cluster_no_list:
                try:
                    cluster = next(cluster for cluster in clusters if cluster['Cluster'] == cluster_no)
                    # Load the best grouping of previous level
                    keyword_cluster_folder = os.path.join(folder, 'group_key_phrases', 'keyword_cluster')
                    path = os.path.join(keyword_cluster_folder,
                                        'group_key_phrases_cluster_#' + str(cluster_no) + '.json')
                    key_phrase_groups = pd.read_json(path).to_dict("records")
                    # Load doc_key_phrases
                    path = os.path.join(folder, 'doc_key_phrase',
                                        'doc_key_phrases_cluster_#' + str(cluster_no) + ".json")
                    doc_key_phrases = pd.read_json(path).to_dict("records")
                    # print(doc_key_phrases)
                    # Re-group keyword cluster > 30
                    new_key_phrase_groups = KeywordClusterUtility.run_re_grouping_experiments(cluster_no, self.model,
                                                                                              key_phrase_groups,
                                                                                              doc_key_phrases)
                    cluster['Key-phrases'] = new_key_phrase_groups
                    # Write new key phrase groups to
                    df = pd.DataFrame(new_key_phrase_groups)
                    df = df[['Group', 'NumPhrases', 'Key-phrases', 'NumDocs', 'DocIds', 'score', 'dimension',
                             'min_samples', 'min_cluster_size']]
                    path = os.path.join(keyword_cluster_folder,
                                        'group_key_phrases_cluster_#' + str(cluster_no) + '.csv')
                    df.to_csv(path, encoding='utf-8', index=False)
                    path = os.path.join(keyword_cluster_folder,
                                        'group_key_phrases_cluster_#' + str(cluster_no) + '.json')
                    df.to_json(path, orient="records")
                    print("=== Complete re-grouping the key phrases in cluster #{c_no} ===".format(
                        c_no=cluster_no))
                except Exception as _err:
                    print("Error occurred! {err}".format(err=_err))
                    sys.exit(-1)
                    # write best results of each group

            # Write output to 'cluster_key_phrases_group'
            df = pd.DataFrame(clusters,
                              columns=['Cluster', 'Key-phrases'])
            folder = os.path.join('output', self.args.case_name, self.args.cluster_folder,
                                  'key_phrases', 'group_key_phrases')
            Path(folder).mkdir(parents=True, exist_ok=True)
            path = os.path.join(folder, 'cluster_key_phrases_group.csv')
            df.to_csv(path, encoding='utf-8', index=False)
            path = os.path.join(folder, 'cluster_key_phrases_group.json')
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
            path = os.path.join(folder, 'key_phrases', 'group_key_phrases', 'cluster_key_phrases_group.json')
            key_phrase_df = pd.read_json(path)
            cluster_df['KeyPhrases'] = key_phrase_df['Key-phrases'].tolist()
            # Re-order cluster df and Output to csv and json file
            cluster_df = cluster_df[['Cluster', 'Score', 'NumDocs', 'DocIds', 'Terms', 'KeyPhrases']]
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
        kp = KeywordCluster()
        # Extract keyword for each article
        kp.extract_doc_key_phrases_by_similarity_diversity()
        kp.combine_cluster_doc_key_phrases()
        # Extract keyword clusters
        # kp.experiment_group_cluster_key_phrases()
        # kp.group_cluster_key_phrases_with_best_experiments()
        # kp.re_group_key_phrases_within_keyword_cluster()
        # kp.combine_terms_key_phrases_results()
    except Exception as err:
        print("Error occurred! {err}".format(err=err))
