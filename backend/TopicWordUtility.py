import os
import re
from pathlib import Path
from nltk.stem import PorterStemmer
import gensim.downloader as api
from gensim.models import KeyedVectors
import pandas as pd
import numpy as np

# Load function words
from sklearn.metrics.pairwise import cosine_similarity


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
                if similar_word[0].lower() not in topic.lower() and \
                        lemma_word not in topic.lower() and \
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

    # Obtain top N (2-grams or 3-gram) topics of a cluster
    @staticmethod
    def collect_top_n_cluster_topics(cluster, model, stop_words, top_n=10, n_gram_type="2"):
        # Check if the topic words appear in the model
        def is_topic_qualified(topic):
            tokens = topic.split(" ")
            # Check if topic word is a word and not a stop word and exists in the model
            topic_words = list(filter(lambda t: bool(re.search('^[a-z]', t.lower()) and
                                                t.lower() not in stop_words and
                                                t.lower() in model), tokens))
            if len(tokens) == len(topic_words):
                return True
            return False
        n_gram_topics = cluster['Topic'+n_gram_type+'-gram'][:50]
        # Get top 10 cluster topics
        cluster_topics = list(filter(lambda w: is_topic_qualified(w['topic']), n_gram_topics))[:top_n]
        # Return a list of topic words
        return list(map(lambda t: t['topic'], cluster_topics))


    @staticmethod
    def compute_similarity_matrix_topics(cluster_topics1, cluster_topics2):
        c_no_1 = cluster_topics1['cluster_no']
        c_no_2 = cluster_topics2['cluster_no']
        # Compute the similarity by average vectors
        topic_vectors1 = cluster_topics1['topic_vectors']
        topic_vectors2 = cluster_topics2['topic_vectors']
        # Create a matrix
        similarity_matrix = np.zeros((len(topic_vectors1), len(topic_vectors2)))

        # Iterate the topics in cluster_1
        for i, vector1 in enumerate(topic_vectors1):
            for j, vector2 in enumerate(topic_vectors2):
                sim = cosine_similarity([vector1, vector2])[0][1]
                similarity_matrix[i, j] = sim
                # print("Similarity between {t1} and {t2} = {sim:.2f}".format(t1=cluster_topics1['topics'][i],
                #                                                             t2=cluster_topics2['topics'][j],
                #                                                             sim=sim))
        sim_df = pd.DataFrame(similarity_matrix, index=cluster_topics1['topics'], columns=cluster_topics2['topics'])
        sim_df = sim_df.round(2)     # Round each similarity to 2 decimal
        # Write to out
        path = os.path.join('output', 'topic', 'matrix',
                            "UrbanStudyCorpus" + '_HDBSCAN_similarity_matrix_' +
                            str(c_no_1) + '_' + str(c_no_2) + '.csv')
        sim_df.to_csv(path, encoding='utf-8')

        return similarity_matrix

    @staticmethod
    def compute_similarity_by_words(w1, w2, model):
        # Get the word vector from Word2Vec model
        def get_word_vectors(tokens):
            # Get the word vectors
            vectors = []
            for token in tokens:
                if token in model:
                    vectors.append(model[token.lower()])
                else:
                    print("Cannot find the vector of " + token)
            return vectors

        # Get the word vector of w1
        tokens_1 = w1.split(" ")
        vectors_1 = get_word_vectors(tokens_1)
        tokens_2 = w2.split(" ")
        vectors_2 = get_word_vectors(tokens_2)
        if len(tokens_1) != len(vectors_1) or len(tokens_2) != len(vectors_2):
            print("Can not get the vector of {w}".format(w=w1))
            return

        # Average the word vector
        avg_vector_1 = np.mean(vectors_1, axis=0)
        avg_vector_2 = np.mean(vectors_2, axis=0)
        # Compute the similarity using cosine similarity
        similarity_matrix = cosine_similarity([avg_vector_1, avg_vector_2])
        sim = similarity_matrix[0][1]
        # if abs(similarity_1 - similarity) > 0.01:
        #     print("Different similarity measures")
        print("Similarity between '{w1}' and '{w2}' = {sim:.2f}".format(
            w1=w1, w2=w2, sim=sim))

