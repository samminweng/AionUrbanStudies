import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from nltk.util import ngrams
from nltk.tokenize import sent_tokenize
from nltk.tokenize import word_tokenize
from Utility import Utility


class TopicCreator:

    # Compute the class-level TF-IDF scores for each cluster of documents
    @staticmethod
    def compute_c_tf_idf_score(clustered_documents, total_number_documents):
        try:
            count = CountVectorizer(ngram_range=(2, 2), stop_words="english").fit(clustered_documents)
            t = count.transform(clustered_documents).toarray()
            w = t.sum(axis=1)
            tf = np.divide(t.T, w)
            sum_t = t.sum(axis=0)
            idf = np.log(np.divide(total_number_documents, sum_t)).reshape(-1, 1)  #
            tf_idf = np.multiply(tf, idf)
            return tf_idf, count
        except Exception as err:
            print("Error occurred! {err}".format(err=err))

    # Obtain top collocation per topic
    @staticmethod
    def extract_top_n_words_per_topic(tf_idf, count, docs_per_topic, n=20):
        collocations = count.get_feature_names()
        labels = list(docs_per_topic['Cluster'])
        tf_idf_transposed = tf_idf.T
        indices = tf_idf_transposed.argsort()[:, -n:]
        top_n_words = {label: [(collocations[j], tf_idf_transposed[i][j]) for j in indices[i]][::-1] for i, label in
                       enumerate(labels)}
        return top_n_words

    @staticmethod
    def get_doc_ids_by_topic_words(text_df, doc_ids, topic_word):
        topic_doc_ids = []
        for i, doc in text_df.iterrows():
            try:
                doc_id = doc['DocId']
                if doc_id in doc_ids:
                    text = doc['Title'] + ". " + doc['Abstract']
                    sentences = sent_tokenize(text.lower())
                    sentences = Utility.clean_sentence(sentences)
                    tokenizes = word_tokenize(' '.join(sentences))
                    bi_grams = list(ngrams(tokenizes, 2))
                    bi_grams = list(map(lambda bi_gram: bi_gram[0] + " " + bi_gram[1], bi_grams))
                    # Check if topic word in bi_grams
                    if topic_word in bi_grams:
                        topic_doc_ids.append(doc_id)
            except Exception as err:
                print("Error occurred! {err}".format(err=err))
        return topic_doc_ids
