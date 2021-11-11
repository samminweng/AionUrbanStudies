import os
from pathlib import Path
import nltk
from nltk.stem import PorterStemmer
import gensim.downloader as api
from gensim.models import KeyedVectors


def load_lemma_word_list():
    # Load the lemma.n file to store the mapping of singular to plural nouns
    _lemma_nouns = {}
    path = os.path.join('data', 'lemma.n')
    f = open(path, 'r')
    lines = f.readlines()
    for line in lines:
        _words = line.rstrip().split("->")  # Remove trailing new line char and split by '->'
        _plural_word = _words[1]
        if '.,' in _plural_word:  # Handle multiple plural forms and get the last one as default plural form
            _plural_word = _plural_word.split('.,')[-1]
        singular_word = _words[0]
        _lemma_nouns[_plural_word] = singular_word
    return _lemma_nouns


class TopicWordUtility:
    lemma_nouns = load_lemma_word_list()  # Store the mapping between singular and plural nouns

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
                if lemma_word not in topic:
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
