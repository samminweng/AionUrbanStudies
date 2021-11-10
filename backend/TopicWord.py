import os.path
from argparse import Namespace
import pandas as pd
import logging
import gensim.downloader as api
# Set logging level
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
# Download the word2vec model



# Compute the word vector of cluster topics and find the similar words of each topic
class TopicWord:
    def __init__(self):
        self.args = Namespace(
            case_name='UrbanStudyCorpus',
        )
        # Load the cluster results as dataframe
        path = os.path.join('output', 'cluster', self.args.case_name + "_HDBSCAN_Cluster_topic_words.json")
        try:
            self.cluster_df = pd.read_json(path)
            print(self.cluster_df)
            # Download the word2vec model
            # Model name ref: https://github.com/RaRe-Technologies/gensim-data
            self.model = api.load("glove-wiki-gigaword-300")
        except Exception as err:
            print("Error occurred! {err}".format(err=err))

    # Load pre-trained Standford GloVe word2vec model using gensim library
    # Ref: https://github.com/stanfordnlp/GloVe
    def load_word2vec_model(self):
        print(self.model)


# Main entry
if __name__ == '__main__':
    word_vector = TopicWord()
