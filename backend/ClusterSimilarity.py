import os.path
from argparse import Namespace
from sentence_transformers import SentenceTransformer
from pathlib import Path
import pandas as pd
import logging
from ClusterSimilarityUtility import ClusterSimilarityUtility
import getpass

# Set logging level
from ClusterUtility import ClusterUtility

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
        path = os.path.join('output', 'cluster', self.args.case_name + "_HDBSCAN_Cluster_topic_words.json")
        df = pd.read_json(path)
        self.clusters = df.to_dict("records")  # Convert to a list of dictionaries
        # Load the published paper text data
        path = os.path.join('data', self.args.case_name + '.json')
        corpus_doc_df = pd.read_json(path)
        self.corpus_docs = corpus_doc_df.to_dict("records")
        duplicate_doc_ids = ClusterUtility.scan_duplicate_articles()
        # Remove duplicated doc
        self.corpus_docs = list(filter(lambda d: d['DocId'] not in duplicate_doc_ids, self.corpus_docs))

    # Find top 30 similar papers for each article in a cluster
    def find_top_similar_paper_in_corpus(self, top_k=30):
        # cluster_no_list = [c_no for c_no in range(-1, 23)]
        cluster_no_list = [-1]
        try:
            model = SentenceTransformer(self.args.model_name, cache_folder=sentence_transformers_path,
                                        device=self.args.device)  # Load sentence transformer model
            # Find top 30 similar papers of each paper in a cluster
            for cluster_no in cluster_no_list:
                ClusterSimilarityUtility.find_top_n_similar_title(cluster_no, self.corpus_docs, self.clusters, model,
                                                                  top_k=top_k)
                # ClusterSimilarityUtility.write_to_title_csv_file(cluster_no, top_k=top_k)

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
    tw.find_top_similar_paper_in_corpus()
    # Test the similarity function
    # TopicWordUtility.compute_similarity_by_words('land cover', 'land use', tw.model)
# #
# # Ref: https://github.com/stanfordnlp/GloVe
# def compute_topic_vectors_and_similar_words(self, cluster_no=15):
#     try:
#         # Get cluster 15
#         cluster = list(filter(lambda c: c['Cluster'] == cluster_no, self.clusters))[0]
#         n_gram_results = []
#         n_gram_topics = cluster['TopicN-gram'][:50]
#         # Get top 30 bi-gram topics
#         for n_gram in n_gram_topics:
#             try:
#                 topic = n_gram['topic']
#                 self.model()
#                 # # Get top 100 similar words by vector
#                 # similar_words = self.model.similar_by_vector(topic_vector, topn=100)
#                 # # Filter out no-word, duplicated topic words and stop words from similar words
#                 # similar_words = list(filter(lambda w: not re.search('[^\\w]', w[0].lower()), similar_words))
#                 # similar_words = list(filter(lambda w: w[0].lower() not in self.stop_words, similar_words))
#                 # # Filter out duplicated noun words
#                 # similar_words = TopicWordUtility.filter_duplicated_words(similar_words, topic)
#                 # # Get top 15 similar words
#                 # N = 15
#                 # result = {"topic": topic, 'topic_vector': topic_vector}
#                 # for i, w in enumerate(similar_words[:N]):
#                 #     result['top_' + str(i) + "_similar_word"] = w[0]
#                 #     result['top_' + str(i) + "_similar_word_score"] = float("{:.2f}".format(w[1]))
#                 # n_gram_results.append(result)
#             except Exception as err:
#                 print("Error occurred! {err}".format(err=err))
#         # Write the topic vectors to csv file
#         n_gram_results = n_gram_results[:10]
#         df = pd.DataFrame(n_gram_results)
#         df['topic_vector'] = df['topic_vector'].apply(lambda v: np.array2string(v, precision=3,
#                                                                                    formatter={'float_kind': lambda
#                                                                                               x: "%.2f" % x}))
#         path = os.path.join('output', 'topic', 'topic_vector',
#                             self.args.case_name + '_' + self.args.approach + '_topic_vector_similar_words_' +
#                             str(cluster_no) + '.csv')
#         df.to_csv(path, encoding='utf-8', index=False)
#     except Exception as err:
#         print("Error occurred! {err}".format(err=err))
