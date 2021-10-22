import os
import nltk
import numpy as np
from matplotlib import pyplot as plt
from nltk import BigramCollocationFinder
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
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
    _df = pd.read_csv(os.path.join('data', 'Function_Words.csv'))
    function_words = _df['Function Word'].tolist()
    # Image path
    image_path = os.path.join('images', 'cluster')
    # Output path
    output_path = os.path.join('output', 'cluster')
    # TF-IDF Term path
    term_path = os.path.join('output', 'term')

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
        fig, (ax0, ax1, ax2) = plt.subplots(1, 3)
        # # Visualise KMeans
        clusterers = cluster_result_df.loc[cluster_result_df['KMeans_Cluster'] != -1, :]
        ax0.scatter(clusterers.x, clusterers.y, c=clusterers['KMeans_Cluster'], linewidth=0, s=5.0, cmap='Spectral')
        ax0.set_title('KMeans')
        # # Visualise HDBScan Outliers
        outliers = cluster_result_df.loc[cluster_result_df['HDBSCAN_Cluster'] == -1, :]
        clusterers = cluster_result_df.loc[cluster_result_df['HDBSCAN_Cluster'] != -1, :]
        max_cluster_number = max(clusterers['HDBSCAN_Cluster'])
        ax1.scatter(outliers.x, outliers.y, color='red', linewidth=0, s=5.0, marker="X")
        ax1.scatter(clusterers.x, clusterers.y, color='gray', linewidth=0, s=5.0)
        ax1.set_title('Outliers (Red)')
        # # Visualise clustered dots using HDBScan
        ax2.scatter(clusterers.x, clusterers.y, c=clusterers['HDBSCAN_Cluster'], linewidth=0, s=5.0, cmap='Spectral')
        ax2.set_title('HDBSCAN')
        #

        # # # Visualise clustered dots using agglomerative
        # clusterers = cluster_result_df.loc[cluster_result_df['Agglomerative_Cluster'] != -1, :]
        # max_clusters = cluster_result_df['Agglomerative_Cluster'].max()
        # ax3_scatters = ax3.scatter(clusterers.x, clusterers.y, c=clusterers['Agglomerative_Cluster'],
        #                            label=clusterers['Agglomerative_Cluster'], linewidth=0, s=5.0, cmap='Spectral')
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
        fig.set_size_inches(10, 5)
        fig.savefig(path, dpi=600)
        print("Output image to " + path)

    @staticmethod
    def derive_bag_of_words(doc_ids, doc_texts):
        try:
            vec = CountVectorizer(stop_words=TopicUtility.stop_words)
            bag_of_words = vec.fit_transform(doc_texts)      # Return a bag of words
            # A bag of words is a matrix. Each row is the document. Each column is a word in vocabulary
            # bag_of_words[i, j] is the occurrence of word 'i' in the document 'j'
            bag_of_words_df = pd.DataFrame(bag_of_words.toarray(), columns=vec.get_feature_names())
            bag_of_words_df['doc_id'] = doc_ids
            # print(counts)
            words_freq = []
            # We go through the vocabulary of bag of words where index is the word at bag of words
            for word, index in vec.vocabulary_.items():
                # Collect all the doc ids that contains the words
                selected_rows = bag_of_words_df[bag_of_words_df[word] > 0]
                # Aggregate doc_ids to a list
                word_doc_ids = list()
                for i, row in selected_rows.iterrows():
                    word_doc_ids.append(row['doc_id'])
                words_freq.append({'topic_words': word, 'doc_ids': word_doc_ids, 'score': len(word_doc_ids)})
            # filter the number of words < 5
            filter_words_freq = list(filter(lambda w: w['score'] >= 5, words_freq))
            # Sort the words_freq by score
            sorted_words_freq = sorted(filter_words_freq, key=lambda w: w['score'], reverse=True)

            return sorted_words_freq
        except Exception as err:
            print("Error occurred! {err}".format(err=err))

    @staticmethod
    def extract_terms_by_TFIDF(doc_ids, texts):
        cleaned_texts = list(map(lambda text: TopicUtility.preprocess_text(text), texts))

        # filter_words = TopicUtility.stop_words + TopicUtility.function_words
        # Filter words containing stop words
        vectorizer = TfidfVectorizer(ngram_range=(2, 2), stop_words=None)
        # Compute tf-idf scores for each word in each sentence of the abstract
        vectors = vectorizer.fit_transform(cleaned_texts)
        feature_names = vectorizer.get_feature_names()
        dense = vectors.todense()
        dense_list = dense.tolist()
        dense_dict = pd.DataFrame(dense_list, columns=feature_names).to_dict(orient='records')
        key_terms = list()
        # Collect all the key terms of all the sentences in the text
        for index, dense in enumerate(dense_dict):
            # Sort the terms by score
            filter_list = list(filter(lambda item: item[1] > 0, dense.items()))
            # Filter topics containing stop words
            filter_list = list(
                filter(lambda item: not Utility.check_words(item[0], TopicUtility.stop_words), filter_list))
            # Filter topics containing function words
            filter_list = list(
                filter(lambda item: not Utility.check_words(item[0], TopicUtility.function_words), filter_list))
            sorted_list = list(sorted(filter_list, key=lambda item: item[1], reverse=True))
            # Include the key terms
            key_terms.append({
                'doc_id': doc_ids[index],
                'key_terms': list(map(lambda item: item[0], sorted_list))
            })
        return key_terms

    # Obtain the tf-idf terms for each individual document in a cluster
    # Select top 2 key term as the representative topics for
    @staticmethod
    def derive_topic_words_tf_idf(tf_idf_df, doc_ids):
        # Obtain the TF-IDF terms for each individual articles in the clustered documents
        topic_words = []
        for doc_id in doc_ids:
            try:
                # Get the top 2 TF-IDF terms
                doc = tf_idf_df.query("DocId == @doc_id")
                if not doc.empty:
                    doc_key_terms = doc.iloc[0]['HDBSCAN_Cluster_KeyTerms']
                    top_terms = doc_key_terms[:2]
                    for top_term in top_terms:
                        found_topics = [topic for topic in topic_words if topic['topic_words'] == top_term]
                        if len(found_topics) == 0:
                            found_topic = {'topic_words': top_term, 'doc_ids': set()}
                            found_topic['doc_ids'].add(doc_id)
                            topic_words.append(found_topic)
                        else:
                            found_topic = found_topics[0]
                            found_topic['doc_ids'].add(doc_id)
                else:
                    print("Can not find doc id = " + str(doc_id))
            except Exception as err:
                print("Error occurred! {err}".format(err=err))
        # Convert the doc_ids to list type and add the score
        for topic in topic_words:
            topic['doc_ids'] = list(topic['doc_ids'])
            topic['score'] = len(topic['doc_ids'])
        # # Sort the topic_words by score
        sorted_topic_words = sorted(topic_words, key=lambda item: item['score'], reverse=True)
        return sorted_topic_words

    @staticmethod
    def derive_topic_words_using_collocations(associate_measure, doc_ids, doc_texts):
        try:
            # Collect a list of clustered document where each document is a list of tokens
            cluster_docs = []
            # Select the documents from doc_ids
            for doc_id, doc_text in zip(doc_ids, doc_texts):
                tokens = word_tokenize(doc_text)
                cluster_docs.append({"doc_id": doc_id, "tokens": tokens})

            # Create NLTK bigram object
            bigram_measures = nltk.collocations.BigramAssocMeasures()
            # Map the cluster documents to a list of document where each doc is represented with a list of tokens
            documents = list(map(lambda doc: doc['tokens'], cluster_docs))
            # Score and rank the collocations
            finder = BigramCollocationFinder.from_documents(documents)
            # finder.apply_freq_filter(4)
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
                topic_words.append({'topic_words': collocation, 'score': score, 'doc_ids': topic_doc_ids})
            # limit the top 20 topic words
            topic_words = topic_words
            # Sort the topic_words by the number of docs
            topic_words = sorted(topic_words, key=lambda topic_word: len(topic_word), reverse=True)
            return topic_words
        except Exception as err:
            print("Error occurred! {err}".format(err=err))

    # Process and clean the text by converting plural nouns to singular nouns
    # Avoid license sentences
    @staticmethod
    def preprocess_text(text):
        try:
            # Split the text into sentence
            sentences = sent_tokenize(text)
            clean_sentences = []
            # Tokenize the text
            for sentence in sentences:
                # Remove all the license relevant sentences.
                if u"\u00A9" not in sentence.lower() and 'licensee' not in sentence.lower() \
                        and 'copyright' not in sentence.lower() and 'rights reserved' not in sentence.lower():
                    words = word_tokenize(sentence)
                    # Tag the words with part-of-speech tags
                    pos_tags = nltk.pos_tag(words)
                    # Convert plural word to singular
                    singular_words = []
                    for pos_tag in pos_tags:
                        word = pos_tag[0]
                        if pos_tag[1] == 'NNS':
                            singular_word = word.rstrip('s')
                            singular_words.append(singular_word)
                        else:
                            singular_words.append(word)
                    # Merge all the words to a sentence
                    clean_sentences.append(" ".join(singular_words))
            # Merge all the sentences to a text
            clean_text = " ".join(clean_sentences)
            return clean_text
        except Exception as err:
            print("Error occurred! {err}".format(err=err))

    # Compute the class-level TF-IDF scores for each cluster of documents
    @staticmethod
    def compute_c_tf_idf_score(n, doc_texts_per_cluster, total_number_documents):
        try:
            # Aggregate every doc in a cluster as a single text
            clustered_texts = list(map(lambda doc: " ".join(doc), doc_texts_per_cluster))
            clean_texts = [TopicUtility.preprocess_text(text) for text in clustered_texts]
            # Vectorize the clustered doc text
            count = CountVectorizer(ngram_range=(n, n), stop_words="english").fit(clean_texts)
            t = count.transform(clean_texts).toarray()
            w = t.sum(axis=1)
            tf = np.divide(t.T, w)
            sum_t = t.sum(axis=0)
            idf = np.log(np.divide(total_number_documents, sum_t)).reshape(-1, 1)  #
            tf_idf = np.multiply(tf, idf)
            return tf_idf, count
        except Exception as err:
            print("Error occurred! {err}".format(err=err))

    # Get topic word (n_grams) by using c-TF-IDF
    @staticmethod
    def get_n_gram_topic_words(approach, docs_per_cluster, total):
        cluster_labels = docs_per_cluster[approach]
        topic_word_list = []
        for n_gram in [1, 2, 3]:
            # Derive topic words using C-TF-IDF
            tf_idf, count = TopicUtility.compute_c_tf_idf_score(n_gram, docs_per_cluster['Text'],
                                                                total)
            # Top_n_word is a dictionary where key is the cluster no and the value is a list of topic words
            # Get 100 topic words per cluster
            topic_words = TopicUtility.extract_top_n_words_per_cluster(tf_idf, count, cluster_labels)
            topic_word_list.append({
                'n_gram': n_gram,
                'topic_words': topic_words
            })
        # Concatenate all the topic words of 1, 2, 3 grams
        topic_word_mix = {}
        for topic_words in topic_word_list:
            for cluster_no, words in topic_words['topic_words'].items():
                if cluster_no not in topic_word_mix:
                    topic_word_mix[cluster_no] = words
                else:
                    # Concatenate all the topic words
                    exiting_words = topic_word_mix.get(cluster_no)
                    topic_word_mix[cluster_no] = exiting_words + words
        # Sort the words by the score and Limit top 50 words for each cluster
        for cluster_no, words in topic_word_mix.items():
            sorted_words = sorted(words, key=lambda word: word[1], reverse=True)
            topic_word_mix[cluster_no] = sorted_words[:50]
        topic_word_list.append({
            'n_gram': -1,   # -1 indicate the mixed grams
            'topic_words': topic_word_mix
        })
        topic_words_df = pd.DataFrame(topic_word_list, columns=['n_gram', 'topic_words'])
        # Write the results to
        path = os.path.join('output', 'cluster', 'temp', 'UrbanStudyCorpus_' + approach + '_n_topic_words.csv')
        topic_words_df.to_csv(path, encoding='utf-8', index=False)
        # # # Write to a json file
        path = os.path.join('output', 'cluster', 'temp', 'UrbanStudyCorpus_' + approach + '_n_topic_words.json')
        topic_words_df.to_json(path, orient='records')
        return topic_words_df

    # Obtain top 100 topic words ranked by c-tf-idf
    @staticmethod
    def extract_top_n_words_per_cluster(tf_idf, count, clusters, n=50):
        n_grams = count.get_feature_names()
        labels = clusters
        tf_idf_transposed = tf_idf.T
        indices = tf_idf_transposed.argsort()[:, -n:]
        top_n_words = {label: [(n_grams[j], tf_idf_transposed[i][j]) for j in indices[i]][::-1] for i, label in
                       enumerate(labels)}
        return top_n_words

    @staticmethod
    def group_docs_by_topic_words(doc_ids, doc_texts, topic_words_per_cluster):
        try:
            docs_per_topic_words = []
            # For each topic word, find out the document ids that contain the topic word
            for topic_words, score in topic_words_per_cluster:
                doc_ids_per_topic = []
                for doc_id, doc_text in zip(doc_ids, doc_texts):
                    # Convert the document text to bi-grams
                    tokenizes = word_tokenize(doc_text)
                    bi_grams = list(ngrams(tokenizes, 2))
                    bi_grams = list(map(lambda bi_gram: bi_gram[0] + " " + bi_gram[1], bi_grams))
                    # Find if the topic words appear in the bi-grams
                    if topic_words in bi_grams:
                        doc_ids_per_topic.append(doc_id)
                if len(doc_ids_per_topic) > 0:
                    docs_per_topic_words.append({'topic_words': topic_words, 'score': score,
                                                 'doc_ids': doc_ids_per_topic})
            # Sort topic words by score
            sorted_docs_per_topic_words = sorted(docs_per_topic_words, key=lambda t: t['score'], reverse=True)
            return sorted_docs_per_topic_words
        except Exception as err:
            print("Error occurred! {err}".format(err=err))

