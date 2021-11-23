import os.path
from argparse import Namespace
from functools import reduce

import numpy as np
from sentence_transformers import SentenceTransformer
from pathlib import Path
import pandas as pd
import logging
from ClusterSimilarityUtility import ClusterSimilarityUtility
import getpass
from sklearn.feature_extraction.text import CountVectorizer

# Set logging level
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
# Set Sentence Transformer path
sentence_transformers_path = os.path.join('/Scratch', getpass.getuser(), 'SentenceTransformer')
if os.name == 'nt':
    sentence_transformers_path = os.path.join("C:", os.sep, "Users", getpass.getuser(), "SentenceTransformer")
Path(sentence_transformers_path).mkdir(parents=True, exist_ok=True)


# Convert the cluster article title into a vector and find the similar articles in other cluster
class ClusterSimilarity:
    def __init__(self):
        self.args = Namespace(
            case_name='UrbanStudyCorpus',
            approach='HDBSCAN',
            # Model name ref: https://www.sbert.net/docs/pretrained_models.html
            model_name="all-mpnet-base-v2",
            device='cuda'
        )
        # Load the cluster results as dataframe
        path = os.path.join('output', 'cluster', self.args.case_name + "_HDBSCAN_Cluster_TF-IDF_topic_words.json")
        df = pd.read_json(path)
        self.clusters = df.to_dict("records")  # Convert to a list of dictionaries
        # Load the published paper text data
        path = os.path.join('data', self.args.case_name + '.json')
        corpus_df = pd.read_json(path)
        corpus_df['Text'] = corpus_df['Title'] + ". " + corpus_df['Abstract']
        # Load HDBSCAN cluster
        path = os.path.join('output', 'cluster', self.args.case_name + "_clusters.json")
        hdbscan_cluster_df = pd.read_json(path)
        # Update corpus data with hdbscan cluster results
        corpus_df['Cluster'] = hdbscan_cluster_df['HDBSCAN_Cluster']
        self.corpus_docs = corpus_df.to_dict("records")
        duplicate_doc_ids = ClusterSimilarityUtility.scan_duplicate_articles()
        # Remove duplicated doc
        self.corpus_docs = list(filter(lambda d: d['DocId'] not in duplicate_doc_ids, self.corpus_docs))

    # # Use the BERT model to extract long key phrases
    # # Ref: https://towardsdatascience.com/keyword-extraction-with-bert-724efca412ea
    def extract_key_phrases_by_clusters(self):
        # Get the cluster vectors by averaging the vectors of each paper in the cluster
        def get_cluster_vector(_model, _cluster_texts, _candidates, _is_load=False):
            if _is_load:
                _path = os.path.join('output', 'similarity', 'temp', 'Cluster_vector_Candidate_vectors.json')
                _df = pd.read_json(_path)
                _dict = _df.to_dict("records")[0]
                return _dict['cluster_vector'], _dict['candidate_vectors']

            # Compute the cluster vector and key phrase vectors
            _np_vectors = _model.encode(_cluster_texts).numpy()  # Convert the numpy array
            _cluster_vector = np.mean(_np_vectors, axis=0)
            _candidate_vectors = _model.encode(_candidates).numpy()
            # Write the vector data to a json file
            _results = list()
            _results.append({'cluster_vector': _cluster_vector, 'candidate_vectors': _candidate_vectors})
            _df = pd.DataFrame(_results)
            # Create a 'temp' folder
            _folder_path = os.path.join('output', 'similarity', 'temp')
            Path(_folder_path).mkdir(parents=True, exist_ok=True)
            _path = os.path.join('output', 'similarity', 'temp', 'Cluster_vector_Candidate_vectors.json')
            _df.to_json(_path, orient='records')
            print("Output the vector results to " + _path)
            return _cluster_vector, _candidate_vectors

        try:
            cluster_no = 5
            cluster_docs = list(filter(lambda d: d['Cluster'] == cluster_no, self.corpus_docs))
            cluster_df = pd.DataFrame(cluster_docs)
            # Clean the texts to lowercase the words, remove punctuations and convert plural to singular nouns
            cluster_df['Text'] = cluster_df['Text'].apply(lambda text: ClusterSimilarityUtility.clean_sentence(text))
            # Concatenate all the texts in this cluster as the cluster doc
            cluster_texts = cluster_df['Text'].to_list()
            cluster_doc = reduce(lambda a, b: a + b, cluster_texts)
            # Extract phrases candidates using N-gram model
            n_gram_range = (3, 3)
            count = CountVectorizer(ngram_range=n_gram_range).fit([cluster_doc])
            candidates = count.get_feature_names()
            # Encode cluster_doc and candidates as BERT embedding
            model = SentenceTransformer(self.args.model_name, cache_folder=sentence_transformers_path,
                                        device=self.args.device)
            # Encode cluster doc and keyword candidates into vectors for comparing the similarity
            cluster_vector, candidates_vectors = get_cluster_vector(model, cluster_texts, candidates, _is_load=False)
            print(candidates_vectors)
        except Exception as err:
            print("Error occurred! {err}".format(err=err))

    # Find top 30 similar papers for each article in a cluster
    def find_top_similar_paper_in_corpus(self, top_k=30):
        # cluster_no_list = [c_no for c_no in range(-1, 23)]
        cluster_no_list = [-1]
        try:
            model = SentenceTransformer(self.args.model_name, cache_folder=sentence_transformers_path,
                                        device=self.args.device)  # Load sentence transformer model
            # # Find top 30 similar papers of each paper in a cluster
            for cluster_no in cluster_no_list:
                ClusterSimilarityUtility.find_top_n_similar_title(cluster_no, self.corpus_docs, self.clusters, model,
                                                                  top_k=top_k)
                # # # Summarize the similar paper results
                ClusterSimilarityUtility.write_to_title_csv_file(cluster_no, top_k=top_k)

            # Collect all the top N topics (words and vectors)
            # cluster_topics = TopicWordUtility.get_cluster_topics(clusters, self.model, top_n=top_n)
            # # # Create a matrix to store cluster similarities
            # cluster_sim_matrix = np.zeros((len(cluster_no_list), len(cluster_no_list)))
            # for i, c1 in enumerate(cluster_no_list):
            #     for j, c2 in enumerate(cluster_no_list):
            #         ct_1 = next(ct for ct in cluster_topics if ct['cluster_no'] == c1)
            #         ct_2 = next(ct for ct in cluster_topics if ct['cluster_no'] == c2)
            #         # Compute the similarity of cluster vectors by using cosine similarity
            #         cv_1 = ct_1['cluster_vectors']
            #         cv_2 = ct_2['cluster_vectors']
            #         sim = cosine_similarity([cv_1, cv_2])[0][1]
            #         cluster_sim_matrix[i, j] = sim
            #         # compute the similarity matrix of cluster #15 and cluster #16 topics
            #         matrix = TopicWordUtility.compute_similarity_matrix_topics(ct_1, ct_2)
            #         # matrix_mean = matrix.mean()
            #         # print("Similarity matrix between cluster #{c1} and #{c2} = {sum}".format(
            #         #     c1=c1, c2=c2, sum=matrix_mean))
            #         # cluster_sim_matrix[i, j] = matrix_mean
            # # rite out cluster similarity matrix
            # df = pd.DataFrame(cluster_sim_matrix, index=cluster_no_list, columns=cluster_no_list)
            # df = df.round(3)  # Round each similarity to 3 decimal
            # # Write to out
            # path = os.path.join('output', 'topic',
            #                     self.args.case_name + '_HDBSCAN_cluster_vector_similarity.csv')
            # df.to_csv(path, encoding='utf-8')
            # # Write JSON path
            # path = os.path.join('output', 'topic',
            #                     self.args.case_name + '_HDBSCAN_cluster_vector_similarity.json')
            # df.to_json(path, orient='records')
        except Exception as err:
            print("Error occurred! {err}".format(err=err))


# Main entry
if __name__ == '__main__':
    tw = ClusterSimilarity()
    # tw.find_top_similar_paper_in_corpus()
    tw.extract_key_phrases_by_clusters()
