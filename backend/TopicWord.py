import os.path
from argparse import Namespace
import pandas as pd
import logging
from gensim.models import KeyedVectors

# Set logging level
from TopicWordUtility import TopicWordUtility

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)


# Download the word2vec model


# Compute the word vector of cluster topics and find the similar words of each topic
class TopicWord:
    def __init__(self):
        self.args = Namespace(
            case_name='UrbanStudyCorpus',
            # Model name ref: https://github.com/RaRe-Technologies/gensim-data
            model_name="glove-wiki-gigaword-50"  # small model for developing
            # model_name = "glove-wiki-gigaword-300"        # Larger GloVe model
            # model_name = "word2vec-google-news-300"       # Google news model
        )
        # Load the cluster results as dataframe
        path = os.path.join('output', 'cluster', self.args.case_name + "_HDBSCAN_Cluster_topic_words.json")
        self.cluster_df = pd.read_json(path)
        self.wv = TopicWordUtility.obtain_keyed_vectors(self.args.model_name, is_load=True)
        vector = self.wv['cat']    # Get a numpy of vectors
        print(vector)

    #
    # Ref: https://github.com/stanfordnlp/GloVe
    def compute_topics(self):
        try:
            # compute cosine similarity
            score = self.wv.similarity('france', 'spain')
            print(score)
        except Exception as err:
            print("Error occurred! {err}".format(err=err))


# Main entry
if __name__ == '__main__':
    topic_word = TopicWord()
    topic_word.compute_topics()
