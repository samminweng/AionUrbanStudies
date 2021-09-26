import os
import nltk
import numpy as np
from matplotlib import pyplot as plt
from nltk import BigramCollocationFinder
from sklearn.feature_extraction.text import CountVectorizer
from nltk.util import ngrams
from nltk.tokenize import sent_tokenize
from nltk.tokenize import word_tokenize
from Utility import Utility
from nltk.corpus import stopwords
import pandas as pd

path = os.path.join('/Scratch', 'mweng', 'nltk_data')
nltk.data.path.append(path)


# Utility for deriving the topics from each cluster of documents.
class TopicUtility:
    case_name = 'UrbanStudyCorpus'
    # Static variable
    stop_words = list(stopwords.words('english'))
    # Load function words
    df = pd.read_csv(os.path.join('data', 'Function_Words.csv'))
    function_words = df['Function Word'].tolist()
    # Image path
    image_path = os.path.join('images', 'cluster')
    # Output path
    output_path = os.path.join('output', 'cluster')

    # # Use 'elbow method' to vary cluster number for selecting an optimal K value
    # # The elbow point of the curve is the optimal K value
    @staticmethod
    def visual_KMean_results(sse_df):
        try:
            fig, ax = plt.subplots()
            # data_points = sse_df.query('n_neighbour == @n_neighbour')
            sse_values = sse_df['sse'].tolist()[:150]
            clusters = sse_df['cluster'].tolist()[:150]
            ax.plot(clusters, sse_values)
            ax.set_xlim(0, 100)
            ax.set_ylim(0, 2500)
            ax.set_xticks(np.arange(0, 101, 5))
            ax.set_xlabel('Number of Clusters')
            ax.set_ylabel('Sum of Square Distances')
            ax.set_title('KMean Value Curve')
            ax.scatter(5, round(sse_values[5]), marker="x")
            ax.scatter(10, round(sse_values[10]), marker="x")
            ax.scatter(15, round(sse_values[15]), marker="x")
            ax.scatter(20, round(sse_values[20]), marker="x")
            # plt.grid(True)
            fig.show()
            path = os.path.join(TopicUtility.image_path, "elbow_curve.png")
            fig.savefig(path)
        except Exception as err:
            print("Error occurred! {err}".format(err=err))

    @staticmethod
    def visualise_cluster_results(min_cluster_size):
        path = os.path.join(TopicUtility.output_path, TopicUtility.case_name + '_clusters.json')
        cluster_result_df = pd.read_json(path)
        fig, ((ax0, ax1), (ax2, ax3)) = plt.subplots(2, 2)
        # # Visualise HDBScan Outliers
        outliers = cluster_result_df.loc[cluster_result_df['HDBSCAN_Cluster'] == -1, :]
        clusterers = cluster_result_df.loc[cluster_result_df['HDBSCAN_Cluster'] != -1, :]
        max_cluster_number = max(clusterers['HDBSCAN_Cluster'])
        ax0.scatter(outliers.x, outliers.y, color='red', linewidth=0, s=5.0, marker="X")
        ax0.scatter(clusterers.x, clusterers.y, color='gray', linewidth=0, s=5.0)
        ax0.set_title('Outliers (Red)')
        # # Visualise clustered dots using HDBScan
        ax1.scatter(clusterers.x, clusterers.y, c=clusterers['HDBSCAN_Cluster'], linewidth=0, s=5.0, cmap='Spectral')
        ax1.set_title('HDBSCan')
        #
        # # Visualise KMeans
        clusterers = cluster_result_df.loc[cluster_result_df['KMeans_Cluster'] != -1, :]
        ax2.scatter(clusterers.x, clusterers.y, c=clusterers['KMeans_Cluster'], linewidth=0, s=5.0, cmap='Spectral')
        ax2.set_title('KMeans')
        # # Visualise clustered dots using agglomerative
        clusterers = cluster_result_df.loc[cluster_result_df['Agglomerative_Cluster'] != -1, :]
        max_clusters = cluster_result_df['Agglomerative_Cluster'].max()
        ax3_scatters = ax3.scatter(clusterers.x, clusterers.y, c=clusterers['Agglomerative_Cluster'],
                                   label=clusterers['Agglomerative_Cluster'], linewidth=0, s=5.0, cmap='Spectral')
        # box = ax3.get_position()
        # ax3.set_position([box.x0, box.y0, box.width * 0.9, box.height])
        #
        # # Produce the legends
        # ax3.legend(*ax3_scatters.legend_elements(), loc='center left', bbox_to_anchor=(1, 0.5))
        #
        # ax3.set_title('agglomerative')
        # # ax3.legend()
        path = os.path.join(TopicUtility.image_path, "cluster_" + str(max_cluster_number) + "_outlier_"
                            + str(len(outliers)) + "_min_cluster_size_" + str(min_cluster_size) + ".png")
        fig.savefig(path, dpi=1200)
        print("Output image to " + path)

    @staticmethod
    def derive_topic_words_using_collocations(associate_measure, doc_ids, doc_texts):
        try:
            # Collect a list of clustered document where each document is a list of tokens
            cluster_docs = []
            # Select the documents from doc_ids
            for index, doc_text in enumerate(doc_texts):
                tokens = word_tokenize(doc_text)
                cluster_docs.append({"doc_id": doc_ids[index], "tokens": tokens})

            # Create NLTK bigram object
            bigram_measures = nltk.collocations.BigramAssocMeasures()
            # Map the cluster documents to a list of document where each doc is represented with a list of tokens
            documents = list(map(lambda doc: doc['tokens'], cluster_docs))
            # Score and rank the collocations
            finder = BigramCollocationFinder.from_documents(documents)
            finder.apply_freq_filter(4)
            # # # Filter out bi_grams containing stopwords or function words
            finder.apply_ngram_filter(lambda w1, w2: w1.lower() in TopicUtility.function_words or
                                                     w2.lower() in TopicUtility.function_words)
            finder.apply_ngram_filter(lambda w1, w2: w1.lower() in TopicUtility.stop_words or
                                                     w2.lower() in TopicUtility.stop_words)
            # Find a list of bi_grams by likelihood collocations
            if associate_measure == 'pmi':
                scored_bi_grams = finder.score_ngrams(bigram_measures.pmi)
            elif associate_measure == 'chi':
                scored_bi_grams = finder.score_ngrams(bigram_measures.chi_sq)
            else:  # likelihood
                scored_bi_grams = finder.score_ngrams(bigram_measures.likelihood_ratio)

            # Sort bi_grams by scores from high to low
            sorted_bi_grams = sorted(scored_bi_grams, key=lambda bi_gram: bi_gram[1], reverse=True)
            # Convert bi_gram object to a list of
            bi_grams_list = list(map(lambda bi_gram: {'collocation': bi_gram[0][0] + " " + bi_gram[0][1],
                                                      'score': bi_gram[1]}, sorted_bi_grams))
            # Collect the doc ids that each collocation appears
            topic_words = []
            for bi_gram in bi_grams_list:
                collocation = bi_gram['collocation']
                score = bi_gram['score']
                topic_doc_ids = []
                for doc in cluster_docs:
                    doc_id = doc['doc_id']
                    doc_tokens = doc['tokens']
                    doc_bi_grams = list(ngrams(doc_tokens, 2))
                    doc_bi_grams = list(map(lambda b: b[0] + " " + b[1], doc_bi_grams))
                    # Check if topic word in bi_grams
                    if collocation in doc_bi_grams:
                        topic_doc_ids.append(doc_id)
                topic_words.append({'collocation': collocation, 'score': score, 'doc_ids': topic_doc_ids})
            # limit the top 10 topic words
            topic_words = topic_words[:30]
            # Sort the topic_words by the number of docs
            topic_words = sorted(topic_words, key=lambda topic_word: len(topic_word), reverse=True)
            return topic_words
        except Exception as err:
            print("Error occurred! {err}".format(err=err))

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
