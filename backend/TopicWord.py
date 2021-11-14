import os.path
import re
from argparse import Namespace
from itertools import combinations, permutations
from pathlib import Path
import numpy as np
import nltk
import pandas as pd
import logging
from nltk.corpus import stopwords
from TopicWordUtility import TopicWordUtility
from sklearn.metrics.pairwise import cosine_similarity

# Set logging level
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
nltk_path = os.path.join("C:", os.sep, "Users", "sam", "nltk_data")
Path(nltk_path).mkdir(parents=True, exist_ok=True)
nltk.data.path.append(nltk_path)
nltk.download('punkt', download_dir=nltk_path)
# Download all the necessary NLTK data
nltk.download('stopwords', download_dir=nltk_path)


# Compute the word vector of cluster topics and find the similar words of each topic
class TopicWord:
    def __init__(self):
        self.args = Namespace(
            case_name='UrbanStudyCorpus',
            approach='HDBSCAN',
            # Model name ref: https://github.com/RaRe-Technologies/gensim-data
            # model_name="glove-wiki-gigaword-50"  # small model for developing
            model_name="glove-wiki-gigaword-300"  # Larger GloVe model
            # model_name="word2vec-google-news-300"       # Google news model
        )
        # Load the cluster results as dataframe
        path = os.path.join('output', 'cluster', self.args.case_name + "_HDBSCAN_Cluster_topic_words.json")
        df = pd.read_json(path)
        self.clusters = df.to_dict("records")  # Convert to a list of dictionaries
        self.model = TopicWordUtility.obtain_keyed_vectors(self.args.model_name, is_load=True)
        # self.model = api.load(self.args.model_name)
        # Static variable
        self.stop_words = list(stopwords.words('english'))
        # print(vector)

    #
    # Ref: https://github.com/stanfordnlp/GloVe
    def compute_topics(self, cluster_no=15):
        try:
            # Get cluster 15
            cluster = list(filter(lambda c: c['Cluster'] == cluster_no, self.clusters))[0]
            results = []
            for n_gram_type in [2, 3]:
                n_gram_results = []
                n_gram_topics = cluster['Topic' + str(n_gram_type) + '-gram'][:30]
                # Get top 30 bi-gram topics
                for n_gram in n_gram_topics:
                    try:
                        topic = n_gram['topic']
                        topic_words = list(filter(lambda w: w.lower() in self.model, topic.split(" ")))
                        # Check both words must in model
                        if len(topic_words) == n_gram_type:
                            # add up the word vectors
                            vectors = []
                            for _word in topic_words:
                                vectors.append(self.model[_word.lower()])
                            # Add up the vectors
                            # word_vector = np.add.reduce(vectors)
                            # Average the word vector
                            word_vector = np.mean(vectors, axis=0)
                            # Substract the word vectors
                            # word_vector = np.subtract(vectors[0], vectors[1])
                            # if len(vectors) == 3:
                            #     word_vector = np.subtract(word_vector, vectors[2])
                            # Get max of vectors
                            # word_vector = np.amax(vectors, axis=0)
                            # Find top 100 similar words by vector
                            similar_words = self.model.similar_by_vector(word_vector, topn=100)
                            # similar_words_1 = self.model.most_simliar(positive=topic_words, topn=100)
                            # Filter out no-word, duplicated topic words and stop words from similar words
                            similar_words = list(filter(lambda w: not re.search('[^\\w]', w[0].lower()), similar_words))
                            similar_words = list(filter(lambda w: w[0].lower() not in self.stop_words, similar_words))
                            # Filter out duplicated noun words
                            similar_words = TopicWordUtility.filter_duplicated_words(similar_words, topic)
                            # Get top 15 similar words
                            N = 15
                            result = {"topic": topic}
                            for i, w in enumerate(similar_words[:N]):
                                result['top_' + str(i) + "_similar_word"] = w[0]
                                result['top_' + str(i) + "_similar_word_score"] = float("{:.2f}".format(w[1]))
                            n_gram_results.append(result)
                    except Exception as err:
                        print("Error occurred! {err}".format(err=err))
                # List top 10 n-gram topic
                results += n_gram_results[:10]
            df = pd.DataFrame(results)
            path = os.path.join('output', 'topic', 'similar_words',
                                self.args.case_name + '_' + self.args.approach + '_topic_similar_words_' +
                                str(cluster_no) + '.csv')
            df.to_csv(path, encoding='utf-8', index=False)
        except Exception as err:
            print("Error occurred! {err}".format(err=err))

    # Measure the similarity of cluster topics
    # Ref: https://radimrehurek.com/gensim/models/keyedvectors.html#what-can-i-do-with-word-vectors
    def compute_similarity(self):
        top_n = 10  # Number of top topics
        cluster_no_list = [15, 16, 12, 7]
        try:
            clusters = list(filter(lambda c: c['Cluster'] in cluster_no_list, self.clusters))
            # Collect all the top N topics (words and vectors)
            cluster_topics = TopicWordUtility.get_cluster_topics(clusters, self.model, top_n=top_n)
            # # Create a matrix to store cluster similarities
            cluster_sim_matrix = np.zeros((len(cluster_no_list), len(cluster_no_list)))
            for i, c1 in enumerate(cluster_no_list):
                for j, c2 in enumerate(cluster_no_list):
                    ct_1 = next(ct for ct in cluster_topics if ct['cluster_no'] == c1)
                    ct_2 = next(ct for ct in cluster_topics if ct['cluster_no'] == c2)
                    # Compute the similarity of cluster vectors by using cosine similarity
                    cv_1 = ct_1['cluster_vectors']
                    cv_2 = ct_2['cluster_vectors']
                    sim = cosine_similarity([cv_1, cv_2])[0][1]
                    cluster_sim_matrix[i, j] = sim
                    # compute the similarity matrix of cluster #15 and cluster #16 topics
                    # matrix = TopicWordUtility.compute_similarity_matrix_topics(ct_1, ct_2)
                    # matrix_mean = matrix.mean()
                    # print("Similarity matrix between cluster #{c1} and #{c2} = {sum}".format(
                    #     c1=c1, c2=c2, sum=matrix_mean))
                    # cluster_sim_matrix[i, j] = matrix_mean
            # rite out cluster similarity matrix
            sim_df = pd.DataFrame(cluster_sim_matrix, index=cluster_no_list, columns=cluster_no_list)
            sim_df = sim_df.round(2)  # Round each similarity to 2 decimal
            # Write to out
            path = os.path.join('output', 'topic',
                                "UrbanStudyCorpus" + '_HDBSCAN_cluster_vector_similarity.csv')
            sim_df.to_csv(path, encoding='utf-8')
        except Exception as err:
            print("Error occurred! {err}".format(err=err))


# Main entry
if __name__ == '__main__':
    tw = TopicWord()
    # # tw.compute_topics(cluster_no=0)
    tw.compute_similarity()
    # Test the similarity function
    # TopicWordUtility.compute_similarity_by_words('land cover', 'land use', tw.model)
    # TopicWordUtility.get_gensim_info()
