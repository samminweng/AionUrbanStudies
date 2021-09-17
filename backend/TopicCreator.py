import numpy as np
from sklearn.feature_extraction.text import CountVectorizer


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
            idf = np.log(np.divide(total_number_documents, sum_t)).reshape(-1, 1)    #
            tf_idf = np.multiply(tf, idf)
            return tf_idf, count
        except Exception as err:
            print("Error occurred! {err}".format(err=err))


    # def extract_top_n_words_per_topics