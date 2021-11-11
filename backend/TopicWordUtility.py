import csv
import os
from functools import reduce
from pathlib import Path
import nltk
from nltk.stem import PorterStemmer
import gensim.downloader as api
from gensim.models import KeyedVectors
import pandas as pd


# Load function words
def load_function_word_list():
    # Load the lemma.n file to store the mapping of singular to plural nouns
    _df = pd.read_csv(os.path.join('data', 'Function_Words.csv'))
    function_words = _df['Function Word'].tolist()
    return function_words


class TopicWordUtility:
    function_words = load_function_word_list()  # Store the mapping between singular and plural nouns

    # # Convert plural word (multiple words) to singular noun
    @staticmethod
    def filter_duplicated_words(similar_words, topic):
        print("Topic = " + topic)
        ps = PorterStemmer()
        filter_similar_words = []
        for similar_word in similar_words:
            try:
                # Lemmatize word:  Convert the plural noun to singular noun and convert the verb to its original type
                lemma_word = ps.stem(similar_word[0].lower())
                # pos_tag = nltk.pos_tag([similar_word[0].lower()])
                # Check if the word is not a substring of topic and the word is a noun
                if similar_word[0].lower() not in topic.lower() and\
                   lemma_word not in topic.lower() and\
                   lemma_word not in TopicWordUtility.function_words:
                    # Check if the word exists
                    exist_words = list(filter(lambda w: ps.stem(w[0].lower()) == lemma_word, filter_similar_words))
                    if len(exist_words) == 0:
                        filter_similar_words.append(similar_word)
            except Exception as err:
                print("Error occurred! {err}".format(err=err))
        return filter_similar_words

    # Obtain the model from gensim data repo and convert to keyed vectors for fast execution
    @staticmethod
    def obtain_keyed_vectors(model_name, is_load=False):

        Path("model").mkdir(parents=True, exist_ok=True)  # Create a model path
        model_path = os.path.join('model', model_name + '.kv')
        if is_load:
            # Load keyed vector model and return
            return KeyedVectors.load(model_path, mmap='r')
        try:
            # Download or load pre-trained Standford GloVe word2vec model using gensim library
            # Gensim library: https://radimrehurek.com/gensim/
            model = api.load(model_name)
            print(model.most_similar("cat"))
            # Save the model to 'model' path
            model.save(model_path)
            return KeyedVectors.load(model_path, mmap='r')
        except Exception as err:
            print("Error occurred! {err}".format(err=err))

