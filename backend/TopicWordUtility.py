import os
import tempfile
from pathlib import Path

import gensim.downloader as api
from gensim.models import KeyedVectors


class TopicWordUtility:
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
