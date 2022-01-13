import math
import os
import re
import logging
import string
from functools import reduce
from pathlib import Path

import hdbscan
import inflect
import nltk
import numpy as np
from matplotlib import pyplot as plt
from nltk import WordNetLemmatizer, pos_tag
from nltk.util import ngrams
from nltk.tokenize import sent_tokenize
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import getpass
import pandas as pd
import plotly.graph_objects as go
import plotly.io as pio
import seaborn as sns
from sklearn.metrics import silhouette_score, pairwise_distances

# Set logging level
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
# Set NLTK data path
nltk_path = os.path.join('/Scratch', getpass.getuser(), 'nltk_data')
if os.name == 'nt':
    nltk_path = os.path.join("C:", os.sep, "Users", getpass.getuser(), "nltk_data")
# Download all the necessary NLTK data
nltk.download('punkt', download_dir=nltk_path)
nltk.download('wordnet', download_dir=nltk_path)
nltk.download('stopwords', download_dir=nltk_path)
nltk.download('averaged_perceptron_tagger', download_dir=nltk_path)  # POS tags
# Append NTLK data path
nltk.data.path.append(nltk_path)


# Utility for deriving the topics from each cluster of documents.
class BERTModelDocClusterUtility:
    case_name = 'UrbanStudyCorpus'
    # Static variable
    stop_words = list(stopwords.words('english'))

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
            path = os.path.join('images', "elbow_curve.png")
            fig.savefig(path)
        except Exception as err:
            print("Error occurred! {err}".format(err=err))

    @staticmethod
    def visualise_cluster_results_by_methods():
        _path = os.path.join(BERTModelDocClusterUtility.output_path,
                             BERTModelDocClusterUtility.case_name + '_clusters.json')
        cluster_result_df = pd.read_json(_path)
        fig, (ax0, ax1, ax2) = plt.subplots(1, 3)
        # # Visualise
        clusters = cluster_result_df.loc[cluster_result_df['KMeans_Cluster'] != -1, :]
        ax0.scatter(clusters.x, clusters.y, c=clusters['KMeans_Cluster'], linewidth=0, s=5.0, cmap='Spectral')
        ax0.set_title('KMeans')
        # # Visualise HDBScan Outliers
        outliers = cluster_result_df.loc[cluster_result_df['HDBSCAN_Cluster'] == -1, :]
        clusters = cluster_result_df.loc[cluster_result_df['HDBSCAN_Cluster'] != -1, :]
        max_cluster_number = max(clusters['HDBSCAN_Cluster'])
        ax1.scatter(outliers.x, outliers.y, color='red', linewidth=0, s=5.0, marker="X")
        ax1.scatter(clusters.x, clusters.y, color='gray', linewidth=0, s=5.0)
        ax1.set_title('Outliers (Red)')
        # # Visualise clustered dots using HDBScan
        ax2.scatter(clusters.x, clusters.y, c=clusters['HDBSCAN_Cluster'], linewidth=0, s=5.0, cmap='Spectral')
        ax2.set_title('HDBSCAN')
        _path = os.path.join('images', "cluster_" + str(max_cluster_number) + "_outlier_" + str(len(outliers)) + ".png")
        fig.set_size_inches(10, 5)
        fig.savefig(_path, dpi=600)
        print("Output image to " + _path)

    @staticmethod
    def visualise_cluster_results_by_iteration(iteration, results, folder):
        try:
            df = pd.DataFrame(results)
            total_clusters = df['HDBSCAN_Cluster'].max() + 1
            # Visualise HDBSCAN clustering results using dot chart
            colors = sns.color_palette('tab20', n_colors=total_clusters).as_hex()
            marker_size = 8
            # Plot clustered dots and outliers
            fig = go.Figure()
            for cluster_no in range(0, total_clusters):
                dots = df.loc[df['HDBSCAN_Cluster'] == cluster_no, :]
                if len(dots) > 0:
                    marker_color = colors[cluster_no]
                    marker_symbol = 'circle'
                    name = 'Cluster {no}'.format(no=cluster_no)
                    fig.add_trace(go.Scatter(
                        name=name,
                        mode='markers',
                        x=dots['x'].tolist(),
                        y=dots['y'].tolist(),
                        marker=dict(line_width=1, symbol=marker_symbol,
                                    size=marker_size, color=marker_color)
                    ))
            # Add outliers
            outliers = df.loc[df['HDBSCAN_Cluster'] == -1, :]
            if len(outliers) > 0:
                fig.add_trace(go.Scatter(
                    name='Outlier',
                    mode='markers',
                    x=outliers['x'].tolist(),
                    y=outliers['y'].tolist(),
                    marker=dict(line_width=1, symbol='x',
                                size=2, color='gray', opacity=0.3)
                ))

            title = 'Iteration = ' + str(iteration)
            # Figure layout
            fig.update_layout(title=title,
                              width=600, height=800,
                              legend=dict(orientation="v"),
                              margin=dict(l=20, r=20, t=30, b=40))
            file_path = os.path.join(folder, 'iteration_' + str(iteration) + ".png")
            pio.write_image(fig, file_path, format='png')
            print("Output the images of clustered results at iteration {i} to {path}".format(i=iteration, path=file_path))
        except Exception as err:
            print("Error occurred! {err}".format(err=err))
    # Visualise the clusters of HDBSCAN by different cluster no
    @staticmethod
    def visualise_cluster_results(cluster_labels, x_pos_list, y_pos_list, parameter, folder):
        try:
            max_cluster_no = max(cluster_labels)
            df = pd.DataFrame()
            df['cluster'] = cluster_labels
            df['x'] = x_pos_list
            df['y'] = y_pos_list
            # Visualise HDBSCAN clustering results using dot chart
            colors = sns.color_palette('tab10', n_colors=max_cluster_no + 1).as_hex()
            marker_size = 8
            # Plot clustered dots and outliers
            fig = go.Figure()
            for cluster_no in range(0, max_cluster_no + 1):
                dots = df.loc[df['cluster'] == cluster_no, :]
                if len(dots) > 0:
                    marker_color = colors[cluster_no]
                    marker_symbol = 'circle'
                    name = 'Cluster {no}'.format(no=cluster_no)
                    fig.add_trace(go.Scatter(
                        name=name,
                        mode='markers',
                        x=dots['x'].tolist(),
                        y=dots['y'].tolist(),
                        marker=dict(line_width=1, symbol=marker_symbol,
                                    size=marker_size, color=marker_color)
                    ))
            # Add outliers
            outliers = df.loc[df['cluster'] == -1, :]
            if len(outliers) > 0:
                fig.add_trace(go.Scatter(
                    name='Outlier',
                    mode='markers',
                    x=outliers['x'].tolist(),
                    y=outliers['y'].tolist(),
                    marker=dict(line_width=1, symbol='x',
                                size=2, color='gray', opacity=0.3)
                ))

            title = 'dimension = ' + str(parameter['dimension']) + ' min samples = ' + str(parameter['min_samples']) + \
                    ' min cluster size =' + str(parameter['min_cluster_size'])
            # Figure layout
            fig.update_layout(title=title,
                              width=600, height=800,
                              legend=dict(orientation="v"),
                              margin=dict(l=20, r=20, t=30, b=40))
            file_name = 'dimension_' + str(parameter['dimension'])
            # Add iteration if needed
            file_name = file_name + ('_iteration_' + str(parameter['iteration']) if 'iteration' in parameter else '')
            file_path = os.path.join(folder, file_name + ".png")
            pio.write_image(fig, file_path, format='png')
            print("Output the images of clustered results to " + file_path)
        except Exception as err:
            print("Error occurred! {err}".format(err=err))

    # Calculate Silhouette score
    # Ref: https://towardsdatascience.com/silhouette-coefficient-validating-clustering-techniques-e976bb81d10c
    # Ref: https://scikit-learn.org/stable/modules/generated/sklearn.metrics.silhouette_score.html
    @staticmethod
    def compute_Silhouette_score(cluster_labels, cluster_vectors):
        # score = 1 indicates good clusters that each cluster distinguishes from other clusters
        # score = 0 no difference between clusters
        # score = -1 clusters are wrong
        try:
            # Get all the cluster dots
            avg_score = silhouette_score(cluster_vectors, cluster_labels, metric='cosine')
            return avg_score
        except Exception as err:
            print("Error occurred! {err}".format(err=err))
            return -1

    # Process and clean the text by converting plural nouns to singular nouns
    # Avoid license sentences
    @staticmethod
    def preprocess_text(text):
        # Change plural nouns to singular nouns using lemmatizer
        def convert_singular_words(_words, _lemmatiser):
            # Tag the words with part-of-speech tags
            _pos_tags = nltk.pos_tag(_words)
            # Convert plural word to singular
            _singular_words = []
            for i, (_word, _pos_tag) in enumerate(_pos_tags):
                try:
                    # NNS indicates plural nouns and convert the plural noun to singular noun
                    if _pos_tag == 'NNS':
                        _singular_word = _lemmatiser.lemmatize(_word.lower())
                        if _word[0].isupper():  # Restore the uppercase
                            _singular_word = _singular_word.capitalize()  # Upper case the first character
                        _singular_words.append(_singular_word)
                    else:
                        _singular_words.append(_word)
                except Exception as _err:
                    print("Error occurred! {err}".format(err=_err))
            # Return all lemmatized words
            return _singular_words

        try:
            lemmatizer = WordNetLemmatizer()
            # Split the text into sentence
            sentences = sent_tokenize(text)
            clean_sentences = []
            # Tokenize the text
            for sentence in sentences:
                # Remove all the license relevant sentences.
                if u"\u00A9" not in sentence.lower() and 'licensee' not in sentence.lower() \
                        and 'copyright' not in sentence.lower() and 'rights reserved' not in sentence.lower():
                    words = word_tokenize(sentence)
                    if len(words) > 0:
                        # Convert plural word to singular
                        singular_words = convert_singular_words(words, lemmatizer)
                        # Merge all the words to a sentence
                        clean_sentences.append(" ".join(singular_words))
            # Merge all the sentences to a text
            clean_text = " ".join(clean_sentences)
            return clean_text
        except Exception as err:
            print("Error occurred! {err}".format(err=err))

    # Generate n-gram candidates from a text (a list of sentences)
    @staticmethod
    def generate_n_gram_candidates(sentences, n_gram_range):
        # Check if n_gram candidate does not have stop words, punctuation or non-words
        def _is_qualified(_n_gram):  # _n_gram is a list of tuple (word, tuple)
            try:
                # qualified_tags = ['NN', 'NNS', 'JJ', 'NNP']
                # # # Check if there is any noun
                nouns = list(filter(lambda _n: _n[1].startswith('NN'), _n_gram))
                if len(nouns) == 0:
                    return False
                # # Check the last word is a nn or nns
                if _n_gram[-1][1] not in ['NN', 'NNS']:
                    return False
                # Check if all words are not stop word or punctuation or non-words
                for _i, _n in enumerate(_n_gram):
                    _word = _n[0]
                    _pos_tag = _n[1]
                    if bool(re.search(r'\d|[^\w]', _word.lower())) or _word.lower() in string.punctuation or \
                            _word.lower() in BERTModelDocClusterUtility.stop_words:  # or _pos_tag not in qualified_tags:
                        return False
                # n-gram is qualified
                return True
            except Exception as _err:
                print("Error occurred! {err}".format(err=_err))

        # Convert n_gram tuples (pos tag and words) to a list of singular words
        def _convert_n_gram_to_words(_n_gram):
            _lemma_words = list()
            for _gram in _n_gram:
                _word = _gram[0]
                _pos_tag = _gram[1]
                _lemma_words.append(_word)
            return " ".join(_lemma_words)

        candidates = list()
        # Extract n_gram from each sentence
        for i, sentence in enumerate(sentences):
            try:
                words = word_tokenize(sentence)
                pos_tags = pos_tag(words)
                # Pass pos tag tuple (word, pos-tag) of each word in the sentence to produce n-grams
                _n_grams = list(ngrams(pos_tags, n_gram_range))
                # Filter out not qualified n_grams that contain stopwords or the word is not alpha_numeric
                for _n_gram in _n_grams:
                    if _is_qualified(_n_gram):
                        n_gram_words = _convert_n_gram_to_words(_n_gram)
                        candidates.append(n_gram_words)  # Convert n_gram (a list of words) to a string
            except Exception as _err:
                print("Error occurred! {err}".format(err=_err))
        return candidates

    # Get topics (n_grams) by using standard TF-IDF and the number of topic is max_length
    @staticmethod
    def get_n_gram_topics(approach, docs_per_cluster_df, folder, is_load=False):
        # A folder that stores all the topic results
        temp_folder = os.path.join(folder, 'temp')
        Path(temp_folder).mkdir(parents=True, exist_ok=True)
        if is_load:
            path = os.path.join(temp_folder, 'TF-IDF_cluster_n_gram_topics.json')
            topic_df = pd.read_json(path)
            topic_list = topic_df.to_dict("records")
            return topic_list

        # Convert the texts of all clusters into a list of document (a list of sentences) to derive n-gram candidates
        def _collect_cluster_docs(_docs_per_cluster_df):
            # Get the clustered texts
            clusters = _docs_per_cluster_df[approach].tolist()
            doc_texts_per_cluster = _docs_per_cluster_df['Text'].tolist()
            _docs = []
            for cluster_no, doc_texts in zip(clusters, doc_texts_per_cluster):
                doc_list = []
                for doc_text in doc_texts:
                    try:
                        if isinstance(doc_text, str):
                            text = BERTModelDocClusterUtility.preprocess_text(doc_text.strip())
                            sentences = sent_tokenize(text)
                            doc_list.extend(sentences)
                    except Exception as _err:
                        print("Error occurred! {err}".format(err=_err))
                _docs.append({'cluster': cluster_no, 'doc': doc_list})  # doc: a list of sentences
            # Convert the frequency matrix to data frame
            df = pd.DataFrame(_docs, columns=['cluster', 'doc'])
            path = os.path.join(temp_folder, 'Step_1_cluster_doc.csv')
            # Write to temp output for validation
            df.to_csv(path, encoding='utf-8', index=False)
            return _docs

        # Create frequency matrix to track the frequencies of a n-gram in
        def _create_frequency_matrix(_docs, _n_gram_range):
            # Vectorized the clustered doc text and Keep the Word case unchanged
            frequency_matrix = []
            for doc in docs:
                cluster_no = doc['cluster']  # doc id is the cluster no
                sentences = doc['doc']
                freq_table = {}
                n_grams = BERTModelDocClusterUtility.generate_n_gram_candidates(sentences, _n_gram_range)
                for n_gram in n_grams:
                    n_gram_text = n_gram.lower()
                    if n_gram_text in freq_table:
                        freq_table[n_gram_text] += 1
                    else:
                        freq_table[n_gram_text] = 1
                frequency_matrix.append({'cluster': cluster_no, 'freq_table': freq_table})
            # Convert the frequency matrix to data frame
            df = pd.DataFrame(frequency_matrix, columns=['cluster', 'freq_table'])
            # Write to temp output for validation
            path = os.path.join(temp_folder, 'Step_2_frequency_matrix.csv')
            df.to_csv(path, encoding='utf-8', index=False)
            return frequency_matrix

        # Compute TF score
        def _compute_tf_matrix(_freq_matrix):
            _tf_matrix = {}
            # Compute tf score for each cluster (doc) in the corpus
            for _row in _freq_matrix:
                _cluster_no = _row['cluster']  # Doc id is the cluster no
                _freq_table = _row['freq_table']  # Store the frequencies of each word in the doc
                _tf_table = {}  # TF score of each word (1,2,3-grams) in the doc
                _total_topics_in_doc = reduce(lambda total, f: total + f, _freq_table.values(), 0)  # Adjusted for total number of words in doc
                for _topic, _freq in _freq_table.items():
                    # frequency of a word in doc / total number of words in doc
                    _tf_table[_topic] = _freq / _total_topics_in_doc
                _tf_matrix[_cluster_no] = _tf_table
            return _tf_matrix

        # Collect the table to store the mapping between word to a list of clusters
        def _create_occs_per_topic(_freq_matrix):
            _occ_table = {}  # Store the mapping between a word and its doc ids
            for _row in _freq_matrix:
                _cluster_no = _row['cluster']  # Doc id is the cluster no
                _freq_table = _row['freq_table']  # Store the frequencies of each word in the doc
                for _topic, _count in _freq_table.items():
                    if _topic in _occ_table:  # Add the table if the word appears in the doc
                        _occ_table[_topic].add(_cluster_no)
                    else:
                        _occ_table[_topic] = {_cluster_no}
            # Convert the doc per word table (a dictionary) to data frame
            _df = pd.DataFrame(list(_occ_table.items()))
            # Write to temp output for validation
            _path = os.path.join(temp_folder, 'Step_3_occs_per_topic.csv')
            _df.to_csv(_path, encoding='utf-8', index=False)
            return _occ_table

        # Compute IDF scores
        def _compute_idf_matrix(_freq_matrix, _occ_per_topic):
            _total_cluster = len(_freq_matrix)  # Total number of clusters in the corpus
            _idf_matrix = {}  # Store idf scores for each doc
            for _row in _freq_matrix:
                _cluster_no = _row['cluster']  # Doc id is the cluster no
                _freq_table = _row['freq_table']  # Store the frequencies of each word in the doc
                _idf_table = {}
                for _topic in _freq_table.keys():
                    _counts = len(_occ_per_topic[_topic])  # Number of clusters the word appears
                    _idf_table[_topic] = math.log10(_total_cluster / float(_counts))
                _idf_matrix[_cluster_no] = _idf_table  # Idf table stores each word's idf scores
            return _idf_matrix

        # Compute tf-idf score matrix
        def _compute_tf_idf_matrix(_tf_matrix, _idf_matrix, _freq_matrix, _occ_per_topic):
            _tf_idf_matrix = {}
            # Compute tf-idf score for each cluster
            for _cluster_no, _tf_table in _tf_matrix.items():
                # Compute tf-idf score of each word in the cluster
                _idf_table = _idf_matrix[_cluster_no]  # idf table stores idf scores of the doc (doc_id)
                # Get freq table of the cluster
                _freq_table = next(f for f in _freq_matrix if f['cluster'] == _cluster_no)['freq_table']
                _tf_idf_list = []
                for _topic, _tf_score in _tf_table.items():  # key is word, value is tf score
                    try:
                        _idf_score = _idf_table[_topic]  # Get idf score of the word
                        _freq = _freq_table[_topic]  # Get the frequencies of the word in cluster doc_id
                        _cluster_ids = sorted(list(_occ_per_topic[_topic]))  # Get the clusters that the word appears
                        _score = float(_tf_score * _idf_score)
                        _tf_idf_list.append({'topic': _topic, 'score': _score, 'freq': _freq,
                                             'cluster_ids': _cluster_ids})
                    except Exception as _err:
                        print("Error occurred! {err}".format(err=_err))
                # Sort tf_idf_list by tf-idf score
                _tf_idf_matrix[str(_cluster_no)] = sorted(_tf_idf_list, key=lambda t: t['score'], reverse=True)
            return _tf_idf_matrix

        # Step 1. Convert each cluster of documents (one or more articles) into a single document
        docs = _collect_cluster_docs(docs_per_cluster_df)
        topics_list = []
        for n_gram_range in [1, 2, 3]:
            try:
                # 2. Create the Frequency matrix of the words in each document (a cluster of articles)
                freq_matrix = _create_frequency_matrix(docs, n_gram_range)
                # 3. Compute Term Frequency (TF) and generate a matrix
                # Term frequency (TF) is the frequency of a word in a document divided by total number of words in the document.
                tf_matrix = _compute_tf_matrix(freq_matrix)
                # 4. Create the table to map the word to a list of documents
                occ_per_topic = _create_occs_per_topic(freq_matrix)
                # 5. Compute IDF (how common or rare a word is) and output the results as a matrix
                idf_matrix = _compute_idf_matrix(freq_matrix, occ_per_topic)
                # Compute tf-idf matrix
                tf_idf_matrix = _compute_tf_idf_matrix(tf_matrix, idf_matrix, freq_matrix, occ_per_topic)
                # Top_n_word is a dictionary where key is the cluster no and the value is a list of topic words
                topics_list.append({
                    'n_gram': n_gram_range,
                    'topics': tf_idf_matrix
                })
            except Exception as err:
                print("Error occurred! {err}".format(err=err))

        topic_df = pd.DataFrame(topics_list, columns=['n_gram', 'topics'])
        # Write the topics results to csv
        topic_df.to_json(os.path.join(temp_folder, 'TF-IDF_cluster_n_gram_topics.json'), orient='records')
        return topics_list  # Return a list of dicts

    # Output the cluster topics extracted by TF-IDF as a csv file
    @staticmethod
    def flatten_tf_idf_topics(cluster_no, folder):
        try:
            path = os.path.join(folder, 'TF-IDF_cluster_topic_n_grams.json')
            cluster_df = pd.read_json(path)
            clusters = cluster_df.to_dict("records")
            cluster = next(cluster for cluster in clusters if cluster['Cluster'] == cluster_no)
            results = []
            for i in range(20):
                result = {'1-gram': "", '1-gram-score': 0, '1-gram-freq': 0, '1-gram-docs': 0, '1-gram-clusters': 0,
                          '2-gram': "", '2-gram-score': 0, '2-gram-freq': 0, '2-gram-docs': 0, '2-gram-clusters': 0,
                          '3-gram': "", '3-gram-score': 0, '3-gram-freq': 0, '3-gram-docs': 0, '3-gram-clusters': 0,
                          'N-gram': "", 'N-gram-score': 0, 'N-gram-freq': 0, 'N-gram-docs': 0, 'N-gram-clusters': 0,
                          }
                for n_gram_num in ['1-gram', '2-gram', '3-gram', 'N-gram']:
                    try:
                        if i < len(cluster['Topic-' + n_gram_num]):
                            n_gram = cluster['Topic-' + n_gram_num][i]
                            result[n_gram_num] = n_gram['topic']
                            result[n_gram_num + '-score'] = n_gram['score']
                            result[n_gram_num + '-freq'] = n_gram['freq']
                            result[n_gram_num + '-docs'] = len(n_gram['doc_ids'])
                            result[n_gram_num + '-clusters'] = len(n_gram['cluster_ids'])
                    except Exception as err:
                        print("Error occurred! {err}".format(err=err))
                results.append(result)
            n_gram_df = pd.DataFrame(results)
            path = os.path.join(folder, 'TF-IDF_cluster_#' + str(cluster_no) + '_flatten_topics.csv')
            n_gram_df.to_csv(path, encoding='utf-8', index=False)
            print('Output topics per cluster to ' + path)
        except Exception as err:
            print("Error occurred! {err}".format(err=err))

    # Group the doc (articles) by individual topic
    @staticmethod
    def group_docs_by_topics(n_gram_range, doc_ids, doc_texts, topics_per_cluster):
        p = inflect.engine()

        # Convert the singular topic into the topic in plural form
        def get_plural_topic_form(_topic):
            # Get plural nouns of topic
            words = _topic.split(" ")
            last_word = words[-1]
            # Get plural word
            plural_word = p.plural(last_word)
            plural_topic = words[:-1] + [plural_word]
            return " ".join(plural_topic)

        try:
            docs_per_topic = []
            # Go through each article and find if each topic appear in the article
            for doc_id, doc_text in zip(doc_ids, doc_texts):
                try:
                    # Convert the preprocessed text to n_grams
                    sentences = sent_tokenize(BERTModelDocClusterUtility.preprocess_text(doc_text))
                    # Obtain the n-grams from the text
                    n_grams = BERTModelDocClusterUtility.generate_n_gram_candidates(sentences, n_gram_range)
                    # For each topic, find out the document ids that contain the topic
                    for item in topics_per_cluster:
                        try:
                            topic = item['topic']
                            score = item['score']
                            freq = item['freq']  # Total number of frequencies in this cluster
                            cluster_ids = item['cluster_ids']  # A list of cluster that topic appears
                            # The topic appears in the article
                            if topic in n_grams:
                                # Check if docs_per_topic contains the doc id
                                doc_topic = next((d for d in docs_per_topic if d['topic'] == topic), None)
                                # Include the doc ids of the topics mentioned in the articles
                                if doc_topic:
                                    doc_topic['doc_ids'].append(doc_id)
                                else:
                                    docs_per_topic.append({'topic': topic, 'score': score, 'freq': freq,
                                                           'cluster_ids': cluster_ids,
                                                           'plural': get_plural_topic_form(topic),
                                                           'doc_ids': [doc_id]})
                        except Exception as err:
                            print("Error occurred! {err}".format(err=err))
                except Exception as err:
                    print("Error occurred! {err}".format(err=err))
            # Sort topics by score
            sorted_docs_per_topics = sorted(docs_per_topic, key=lambda t: t['score'], reverse=True)
            return sorted_docs_per_topics
        except Exception as err:
            print("Error occurred! {err}".format(err=err))

    # Filter the overlapping topics of mix-grams (1, 2, 3) in a cluster, e.g. 'air' and 'air temperature' can be merged
    # if they appear in the same set of articles and the same set of clusters. 'air' topic can be merged to 'air temperature'
    @staticmethod
    def merge_n_gram_topic(n_gram_topics):
        try:
            # Sort n-grams by score
            sorted_n_grams = sorted(n_gram_topics, key=lambda _n_gram: _n_gram['score'], reverse=True)
            duplicate_topics = set()
            # Scan the mixed n_gram_topic and find duplicated topics to another topic
            for n_gram_topic in sorted_n_grams:
                topic = n_gram_topic['topic']
                score = n_gram_topic['score']
                freq = n_gram_topic['freq']
                cluster_ids = set(n_gram_topic['cluster_ids'])  # a set of cluster ids
                doc_ids = set(n_gram_topic['doc_ids'])
                # Scan if any other sub topic have the same freq and cluster_ids and share similar topics
                # The topic (such as 'air') is a substring of another topic ('air temperature') so 'air' is duplicated
                relevant_topics = list(
                    filter(lambda _n_gram: _n_gram['topic'] != topic and topic in _n_gram['topic'] and
                                           _n_gram['freq'] == freq and
                                           len(set(_n_gram['doc_ids']) - doc_ids) == 0 and
                                           len(set(_n_gram['cluster_ids']) - cluster_ids) == 0,
                           sorted_n_grams))
                if len(relevant_topics) > 0:  # We have found other relevant topics that can cover this topic
                    duplicate_topics.add(topic)
            # Removed duplicated topics and single char (such as 'j')
            filter_topics = list(
                filter(lambda _n_gram: len(_n_gram['topic']) > 1 and _n_gram['topic'] not in duplicate_topics,
                       sorted_n_grams))
            # Sort by the score and  The resulting topics are mostly 2 or 3 grams
            filter_sorted_topics = sorted(filter_topics, key=lambda _n_gram: _n_gram['score'], reverse=True)
            return filter_sorted_topics[:300]  # Get top 300 topics
        except Exception as err:
            print("Error occurred! {err}".format(err=err))

    # Experiment HDBSCAN clustering with different kinds of parameters
    @staticmethod
    def cluster_outliers_experiments_by_hdbscan(dimension, doc_vectors):
        # Collect clustering results and find outliers and the cluster of minimal size
        def collect_cluster_results(_results, _cluster_label):
            try:
                _found = next((r for r in _results if r['cluster_no'] == _cluster_label), None)
                if not _found:
                    _results.append({'cluster_no': _cluster_label, 'count': 1})
                else:
                    _found['count'] += 1
                # Sort the results
                _results = sorted(_results, key=lambda c: (c['count'], c['cluster_no']))
                return _results
            except Exception as _err:
                print("Error occurred! {err}".format(err=_err))

        # Store all experiment results
        results = list()
        # Cluster the doc vectors with different parameters
        for min_samples in [None] + list(range(1, 21)):
            for min_cluster_size in range(5, 21):
                for epsilon in [0.0]:
                    result = {'dimension': dimension,
                              'min_cluster_size': min_cluster_size,
                              'min_samples': str(min_samples),
                              'epsilon': epsilon,
                              'outliers': 'None',
                              'total_clusters': 'None',
                              'cluster_results': 'None',
                              'Silhouette_score': 'None',
                              'cluster_labels': 'None'}
                    try:
                        # Compute the cosine distance/similarity for each doc vectors
                        distances = pairwise_distances(doc_vectors, metric='cosine')
                        # Apply UMAP to reduce vectors
                        cluster_labels = hdbscan.HDBSCAN(min_cluster_size=min_cluster_size,
                                                         min_samples=min_samples,
                                                         cluster_selection_epsilon=epsilon,
                                                         metric='precomputed').fit_predict(
                            distances.astype('float64')).tolist()
                        df = pd.DataFrame()
                        df['cluster_labels'] = cluster_labels
                        df['doc_vectors'] = distances.tolist()
                        # Include the cluster labels
                        result['cluster_labels'] = cluster_labels
                        # Collect all the cluster labels as a single list
                        cluster_results = reduce(lambda pre, cur: collect_cluster_results(pre, cur), cluster_labels,
                                                 list())
                        result['cluster_results'] = cluster_results
                        # Compute silhouette score
                        outlier_df = df[df['cluster_labels'] == -1]
                        no_outlier_df = df[df['cluster_labels'] != -1]
                        result['outliers'] = len(outlier_df)
                        result['total_clusters'] = len(cluster_results)
                        if len(no_outlier_df) > 0:
                            score = BERTModelDocClusterUtility.compute_Silhouette_score(
                                no_outlier_df['cluster_labels'].tolist(),
                                np.vstack(no_outlier_df['doc_vectors'].tolist()))
                            result['Silhouette_score'] = score
                    except Exception as err:
                        print("Error occurred! {err}".format(err=err))
                    print(result)
                    results.append(result)

        return results

    # Find the duplicate papers in the corpus
    @staticmethod
    def clean_corpus(case_name):
        # Convert the raw source files downloaded from Scopus
        def _convert_corpus():
            _folder = os.path.join('data', case_name)
            _corpus_df = pd.read_csv(os.path.join(_folder, case_name + '_scopus.csv'))
            _corpus_df['DocId'] = list(range(1, len(_corpus_df) + 1))
            # Select columns
            _corpus_df = _corpus_df[['DocId', 'Cited by', 'Title', 'Author Keywords', 'Abstract', 'Year',
                                     'Source title', 'Authors', 'DOI', 'Document Type']]
            # # Output as csv file
            _corpus_df.to_csv(os.path.join(_folder, case_name + '.csv'),
                              encoding='utf-8', index=False)

        try:
            _convert_corpus()
            folder = os.path.join('data', case_name)
            corpus_df = pd.read_csv(os.path.join(folder, case_name + '.csv'))
            corpus = corpus_df.to_dict("records")
            irrelevant_doc_ids = set()
            # Check if the paper has other identical paper in the corpus
            for article in corpus:
                doc_id = article['DocId']
                title = article['Title']
                # Find if other article has the same title and author names
                identical_articles = list(
                    filter(lambda _article: _article['Title'].lower().strip() == title.lower().strip() and
                                            _article['DocId'] > doc_id, corpus))
                # add the article docs
                for sa in identical_articles:
                    irrelevant_doc_ids.add(sa['DocId'])
            # Get the all ir-relevant docs
            ir_df = corpus_df[corpus_df['DocId'].isin(irrelevant_doc_ids)]
            ir_df = ir_df[['DocId', 'Title', 'Abstract']]
            # Output as csv file
            ir_df.to_csv(os.path.join(folder, case_name + '_irrelevant_docs.csv'),
                         encoding='utf-8', index=False)
            # Get all  outliers
            df = corpus_df[~corpus_df['DocId'].isin(irrelevant_doc_ids)]
            # Save the cleaned df without document vectors into csv and json files
            df_clean = df.copy(deep=True)
            # df_clean.drop(['DocVectors'], inplace=True, axis=1)
            path = os.path.join('data', case_name, case_name + '_cleaned.csv')
            df_clean.to_csv(path, encoding='utf-8', index=False)
            path = os.path.join('data', case_name, case_name + '_cleaned.json')
            df_clean.to_json(path, orient='records')
        except Exception as err:
            print("Error occurred! {err}".format(err=err))
