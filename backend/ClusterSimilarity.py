import itertools
import os.path
import re
from argparse import Namespace

import umap
from sentence_transformers import util
from sentence_transformers import SentenceTransformer
from pathlib import Path
import pandas as pd
import logging
from ClusterSimilarityUtility import ClusterSimilarityUtility
import getpass

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
            device='cpu'
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
        try:
            # # Encode cluster_doc and candidates as BERT embedding
            model = SentenceTransformer(self.args.model_name, cache_folder=sentence_transformers_path,
                                        device=self.args.device)
            for cluster_no in [4]:
                cluster_docs = list(filter(lambda d: d['Cluster'] == cluster_no, self.corpus_docs))[0:4]
                results = list()
                for doc in cluster_docs:
                    doc_id = doc['DocId']
                    # Get the first doc
                    doc = next(doc for doc in cluster_docs if doc['DocId'] == doc_id)
                    sentences = ClusterSimilarityUtility.clean_sentence(doc['Text'])
                    doc_text = " ".join(list(map(lambda s: " ".join(s), sentences)))
                    result = {'Cluster': cluster_no, 'DocId': doc_id}
                    candidates = []
                    for n_gram_range in [1, 2, 3]:
                        try:
                            # Extract key phrase candidates using n-gram
                            n_gram_candidates = ClusterSimilarityUtility.generate_n_gram_candidates(sentences,
                                                                                                    n_gram_range)
                            # Fine top k key phrases similar to a paper
                            # Get the key phrase words only
                            top_n_gram_key_phrases = ClusterSimilarityUtility.get_top_key_phrases(model, doc_text, n_gram_candidates, top_k=30)
                            result[str(n_gram_range) + '-gram-key-phrases'] = top_n_gram_key_phrases
                            candidates = candidates + top_n_gram_key_phrases
                        except Exception as err:
                            print("Error occurred! {err}".format(err=err))
                    # Get all the n-gram key phrases of a doc
                    doc_top_key_phrases = ClusterSimilarityUtility.get_top_key_phrases(model, doc_text, candidates, top_k=10)
                    result['key-phrases'] = doc_top_key_phrases
                    results.append(result)
                # # # Write key phrases to output
                df = pd.DataFrame(results, columns=['Cluster', 'DocId', 'key-phrases', '1-gram-key-phrases',
                                                    '2-gram-key-phrases', '3-gram-key-phrases'])
                folder = os.path.join('output', 'similarity', 'key_phrases')
                Path(folder).mkdir(parents=True, exist_ok=True)
                path = os.path.join(folder, 'mmr_top_key_phrases_cluster_#' + str(cluster_no) + '.csv')
                df.to_csv(path, encoding='utf-8', index=False)
                # Load to a data frame
                # reduce(lambda r1, r2: r1 + r2, list(map(lambda r: r['key-phrase'], results)))



                # key_phrase_vector = model.encode(key_phrase_df['key'])
                # # Reduce the vectors of key
                # reduced_vectors = umap.UMAP(
                #     n_neighbors=100,
                #     min_dist=0.0,
                #     n_components=2,
                #     random_state=42,
                #     metric='cosine'
                # ).fit_transform(key_phrase_df['vector'].to_list())
                # print(reduced_vectors)






                #         # # # Write to a json file
                #         path = os.path.join('output', 'similarity', 'key_phrases',
                #                             'mmr_top_key_phrases_cluster_' + str(cluster_no) +
                #                             '_n_gram_' + str(n_gram_range[0]) + '.json')
                #         df.to_json(path, orient='records')
                #         print('Output key phrases to ' + path)

                #

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
