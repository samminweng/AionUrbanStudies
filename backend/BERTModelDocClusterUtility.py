import math
import os
import re
import logging
import nltk
import numpy as np
from matplotlib import pyplot as plt
from nltk import BigramCollocationFinder
from nltk.util import ngrams
from nltk.tokenize import sent_tokenize
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import getpass
import pandas as pd
import plotly.graph_objects as go
import plotly.io as pio
import seaborn as sns
from sklearn.metrics import silhouette_score

# Set logging level
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
# Set NLTK data path
nltk_path = os.path.join('/Scratch', getpass.getuser(), 'nltk_data')
if os.name == 'nt':
    nltk_path = os.path.join("C:", os.sep, "Users", getpass.getuser(), "nltk_data")
# Download all the necessary NLTK data
nltk.download('punkt', download_dir=nltk_path)
nltk.download('stopwords', download_dir=nltk_path)
# Append NTLK data path
nltk.data.path.append(nltk_path)


# Utility for deriving the topics from each cluster of documents.
class BERTModelDocClusterUtility:
    case_name = 'UrbanStudyCorpus'
    # Static variable
    stop_words = list(stopwords.words('english'))
    # Output path
    output_path = os.path.join('output', 'cluster')
    # Load the lemma.n file to store the mapping of singular to plural nouns
    lemma_nouns = {}
    path = os.path.join('data', 'lemma.n')
    f = open(path, 'r')
    lines = f.readlines()
    for line in lines:
        words = line.rstrip().split("->")  # Remove trailing new line char and split by '->'
        plural_word = words[1]
        if '.,' in plural_word:  # Handle multiple plural forms and get the last one as default plural form
            plural_word = plural_word.split('.,')[-1]
        singular_word = words[0]
        lemma_nouns[plural_word] = singular_word

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

            title = 'dimension = ' + str(parameter['dimension'])
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
            print("Output the cluster results to " + file_path)
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
            return None

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
                    if len(words) > 0:
                        words[0] = words[0].lower()  # Make the 1st word of a sentence lowercase
                        # Tag the words with part-of-speech tags
                        pos_tags = nltk.pos_tag(words)
                        # Convert plural word to singular
                        singular_words = []
                        for pos_tag in pos_tags:
                            word = pos_tag[0]
                            # NNS indicates plural nouns
                            if pos_tag[1] == 'NNS':
                                singular_word = word.rstrip('s')
                                if word in BERTModelDocClusterUtility.lemma_nouns:
                                    singular_word = BERTModelDocClusterUtility.lemma_nouns[word]
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

    # Get topics (n_grams) by using standard TF-IDF and the number of topic is max_length
    @staticmethod
    def get_n_gram_topics(approach, docs_per_cluster, folder, is_load=False):
        # A folder that stores all the topic results
        temp_folder = os.path.join(folder, 'topics', 'temp')
        if is_load:
            n_gram_topics_df = pd.read_json(os.path.join(temp_folder,
                                                         'UrbanStudyCorpus_' + approach + '_n_topics.json'))
            return n_gram_topics_df.to_dict("records")

        # Convert the texts of all clusters into a list of document (a list of sentences) for deriving n-grams
        def _collect_cluster_docs(_docs_per_cluster):
            # Get the clustered texts
            clusters = _docs_per_cluster[approach].tolist()
            doc_texts_per_cluster = docs_per_cluster['Text'].tolist()
            _docs = []
            for cluster_no, doc_texts in zip(clusters, doc_texts_per_cluster):
                doc_list = []
                for doc_text in doc_texts:
                    try:
                        if isinstance(doc_text, str):
                            text = BERTModelDocClusterUtility.preprocess_text(doc_text.strip())
                            sentences = sent_tokenize(text)
                            doc_list.extend(sentences)
                    except Exception as err:
                        print("Error occurred! {err}".format(err=err))
                _docs.append({'cluster': cluster_no, 'doc': doc_list})  # doc: a list of sentences
            # Convert the frequency matrix to data frame
            df = pd.DataFrame(_docs, columns=['cluster', 'doc'])
            # Write to temp output for validation
            df.to_csv(os.path.join(temp_folder, 'Step_1_UrbanStudyCorpus_cluster_doc.csv'),
                      encoding='utf-8', index=False)
            df.to_json(os.path.join(temp_folder, 'Step_1_UrbanStudyCorpus_cluster_doc.json'),
                       orient='records')
            return _docs

        # Create frequency matrix to track the frequencies of a n-gram in
        def _create_frequency_matrix(_docs, _num):
            # Generate n-gram of a text and avoid stop
            def _generate_ngrams(_words, _num):
                _n_grams = list(ngrams(_words, _num))
                # Filter out not qualified n_grams that contain stopwords or the word is not alpha_numeric
                r_list = []
                for _n_gram in _n_grams:
                    is_qualified = True
                    for word in _n_gram:
                        # Each word in 'n-gram' must not be stop words and must be a alphabet or number
                        if word.lower() in BERTModelDocClusterUtility.stop_words or \
                                re.search('\d|[^\w]', word.lower()):
                            is_qualified = False
                            break
                    if is_qualified:
                        r_list.append(" ".join(_n_gram))  # Convert n_gram (a list of words) to a string
                return r_list

            # Vectorized the clustered doc text and Keep the Word case unchanged
            frequency_matrix = []
            for doc in docs:
                cluster_no = doc['cluster']  # doc id is the cluster no
                doc_texts = doc['doc']
                freq_table = {}
                for sent in doc_texts:
                    words = word_tokenize(sent)
                    n_grams = _generate_ngrams(words, _num)
                    for ngram in n_grams:
                        if ngram in freq_table:
                            freq_table[ngram] += 1
                        else:
                            freq_table[ngram] = 1
                frequency_matrix.append({'cluster': cluster_no, 'freq_table': freq_table})
            # Convert the frequency matrix to data frame
            df = pd.DataFrame(frequency_matrix, columns=['cluster', 'freq_table'])
            # Write to temp output for validation
            path = os.path.join(temp_folder, 'Step_2_UrbanStudyCorpus_frequency_matrix.csv')
            df.to_csv(path, encoding='utf-8', index=False)
            path = os.path.join(temp_folder, 'Step_2_UrbanStudyCorpus_frequency_matrix.json')
            df.to_json(path, orient='records')
            print('Output topics per cluster to ' + path)
            return frequency_matrix

        # Compute TF score
        def _compute_tf_matrix(_freq_matrix):
            _tf_matrix = {}
            # Compute tf score for each cluster (doc) in the corpus
            for row in freq_matrix:
                doc_id = row['cluster']  # Doc id is the cluster no
                freq_table = row['freq_table']  # Store the frequencies of each word in the doc
                _tf_table = {}  # TF score of each word (1,2,3-grams) in the doc
                _total_words_in_doc = len(freq_table)  # Adjusted for total number of words in doc
                for word, freq in freq_table.items():
                    # frequency of a word in doc / total number of words in doc
                    _tf_table[word] = freq / _total_words_in_doc
                _tf_matrix[doc_id] = _tf_table
            return _tf_matrix

        # Collect the table to store the mapping between word to a list of documents (clusters)
        def _create_docs_per_word(_freq_matrix):
            word_doc_table = {}  # Store the mapping between a word and its doc ids
            for row in _freq_matrix:
                doc_id = row['cluster']  # Doc id is the cluster no
                freq_table = row['freq_table']  # Store the frequencies of each word in the doc
                for word, count in freq_table.items():
                    if word in word_doc_table:  # Add the table if the word appears in the doc
                        word_doc_table[word].append(doc_id)
                    else:
                        word_doc_table[word] = [doc_id]

                # Convert the doc per word table (a dictionary) to data frame
                df = pd.DataFrame(list(word_doc_table.items()))
                # Write to temp output for validation
                df.to_csv(os.path.join(temp_folder,
                                       'Step_3_UrbanStudyCorpus_word_doc_table.csv'),
                          encoding='utf-8', index=False)
                df.to_json(os.path.join(temp_folder,
                                        'Step_3_UrbanStudyCorpus_word_doc_table.json'),
                           orient='records')
            return word_doc_table

        # Compute IDF scores
        def _compute_idf_matrix(_freq_matrix, _doc_per_words):
            _total_dos = len(_freq_matrix)  # Total number of clusters in the corpus
            _idf_matrix = {}  # Store idf scores for each doc
            for row in _freq_matrix:
                doc_id = row['cluster']  # Doc id is the cluster no
                freq_table = row['freq_table']  # Store the frequencies of each word in the doc
                idf_table = {}
                for word in freq_table.keys():
                    counts = len(_doc_per_words[word])  # Number of documents (clusters) the word appears
                    idf_table[word] = math.log10(_total_dos / float(counts))
                _idf_matrix[doc_id] = idf_table  # Idf table stores each word's idf scores
            return _idf_matrix

        # Compute tf-idf score matrix
        def _compute_tf_idf_matrix(_tf_matrix, _idf_matrix, _freq_matrix, _docs_per_word):
            _tf_idf_matrix = {}
            # Compute tf-idf score for each cluster
            for doc_id, tf_table in _tf_matrix.items():
                # Compute tf-idf score of each word in the cluster
                idf_table = _idf_matrix[doc_id]  # idf table stores idf scores of the doc (doc_id)
                # Get freq table of the cluster
                freq_table = next(f for f in _freq_matrix if f['cluster'] == doc_id)['freq_table']
                tf_idf_list = []
                for word, tf_score in tf_table.items():  # key is word, value is tf score
                    try:
                        idf_score = idf_table[word]  # Get idf score of the word
                        freq = freq_table[word]  # Get the frequencies of the word in cluster doc_id
                        cluster_ids = _docs_per_word[word]  # Get the clusters that the word appears
                        tf_idf_list.append({'topic': word, 'score': float(tf_score * idf_score), 'freq': freq,
                                            'cluster_ids': cluster_ids})
                    except Exception as err:
                        print("Error occurred! {err}".format(err=err))
                # Sort the tf_idf_list
                sorted_tf_idf_list = sorted(tf_idf_list, key=lambda t: t['score'], reverse=True)
                # Store tf-idf scores of the document
                _tf_idf_matrix[str(doc_id)] = sorted_tf_idf_list
            return _tf_idf_matrix

        # Step 1. Convert each cluster of documents (one or more articles) into a single document
        docs = _collect_cluster_docs(docs_per_cluster)
        topics_list = []
        for n_gram_num in [1, 2, 3]:
            try:
                # 2. Create the Frequency matrix of the words in each document (a cluster of articles)
                freq_matrix = _create_frequency_matrix(docs, n_gram_num)
                # 3. Compute Term Frequency (TF) and generate a matrix
                # Term frequency (TF) is the frequency of a word in a document divided by total number of words in the document.
                tf_matrix = _compute_tf_matrix(freq_matrix)
                # 4. Create the table to map the word to a list of documents
                docs_per_word = _create_docs_per_word(freq_matrix)
                # 5. Compute IDF (how common or rare a word is) and output the results as a matrix
                idf_matrix = _compute_idf_matrix(freq_matrix, docs_per_word)
                # Compute tf-idf matrix
                tf_idf_matrix = _compute_tf_idf_matrix(tf_matrix, idf_matrix, freq_matrix, docs_per_word)
                # print(tf_idf_matrix)
                # Top_n_word is a dictionary where key is the cluster no and the value is a list of topic words
                topics_list.append({
                    'n_gram': n_gram_num,
                    'topics': tf_idf_matrix
                })
            except Exception as err:
                print("Error occurred! {err}".format(err=err))

        topic_words_df = pd.DataFrame(topics_list, columns=['n_gram', 'topics'])
        # Write the topics results to csv
        topic_words_df.to_csv(path_or_buf=os.path.join(temp_folder, 'UrbanStudyCorpus_' + approach + '_n_topics.csv'),
                              encoding='utf-8', index=False)
        # # # Write to a json file
        topic_words_df.to_json(os.path.join(temp_folder, 'UrbanStudyCorpus_' + approach + '_n_topics.json'), orient='records')
        return topics_list  # Return a list of dicts

    # Output the cluster topics extracted by TF-IDF as a csv file
    @staticmethod
    def flatten_tf_idf_topics(cluster_no, folder):
        cluster = "HDBSCAN_Cluster"
        approach = "TF-IDF"
        try:
            path = os.path.join(folder, 'UrbanStudyCorpus_' + cluster + '_' + approach + '_topic_words_n_grams.json')
            cluster_df = pd.read_json(path)
            clusters = cluster_df.to_dict("records")
            cluster = next(cluster for cluster in clusters if cluster['Cluster'] == cluster_no)
            results = []
            for i in range(50):
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
            path = os.path.join(folder, 'UrbanStudyCorpus_' + approach + '_cluster_#' + str(cluster_no) + '_flatten_topics.csv')
            n_gram_df.to_csv(path, encoding='utf-8', index=False)
            print('Output topics per cluster to ' + path)
        except Exception as err:
            print("Error occurred! {err}".format(err=err))

    # Group the doc (articles) by individual topic
    @staticmethod
    def group_docs_by_topics(n_gram_num, doc_ids, doc_texts, topics_per_cluster):
        # Convert the singular topic into the topic in plural form
        def get_plural_topic_form(_topic):
            # Get plural nouns of topic
            words = _topic.split(" ")
            last_word = words[-1]
            plural_word = last_word + "s"
            for plural, singular in BERTModelDocClusterUtility.lemma_nouns.items():
                if singular == last_word:
                    plural_word = plural
                    break
            plural_topic = words[:-1] + [plural_word]
            return " ".join(plural_topic)

        try:
            docs_per_topic = []
            # Go through each article and find if each topic appear in the article
            for doc_id, doc_text in zip(doc_ids, doc_texts):
                try:
                    # Convert the preprocessed text to n_grams
                    tokenizes = word_tokenize(BERTModelDocClusterUtility.preprocess_text(doc_text))
                    # Obtain the n-grams from the text
                    n_grams = list(ngrams(tokenizes, n_gram_num))
                    n_grams = list(map(lambda n_gram: " ".join(n_gram), n_grams))
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

    @staticmethod
    def get_outlier_doc_ids(is_load=True):
        if is_load:
            path = os.path.join('data', 'UrbanStudyCorpus_outliers.csv')
            outlier_df = pd.read_csv(path)
            outliers = outlier_df.to_dict("records")
            # Return a list of doc ids
            return list(map(lambda doc: doc['DocId'], outliers))

        try:
            # Read HDBSCAN outlier
            outlier_df = pd.read_csv(os.path.join('output', 'cluster', 'experiments', 'hdbscan', 'HDBSCAN_outlier.csv'))
            outliers = outlier_df.to_dict("records")
            # Scan duplicate doc in the corpus
            corpus_df = pd.read_csv(os.path.join('data', 'UrbanStudyCorpus.csv'))
            corpus = corpus_df.to_dict("records")
            # Check if a doc has the same title in the
            for doc in corpus:
                doc_id = doc['DocId']
                title = doc['Title']
                # Find if other article has the same title
                duplicates_docs = list(filter(lambda a: a['Title'].lower().strip() == title.lower().strip() and
                                                        a['DocId'] > doc_id, corpus))
                for duplicate_doc in duplicates_docs:
                    # Check if the common doc exits in outliers
                    found_doc = next((outlier_doc for outlier_doc in outliers
                                      if outlier_doc['DocId'] == duplicate_doc['DocId']), None)
                    if not found_doc:  # If not, add the duplicate doc to outlier docs
                        outliers.append({'DocId': duplicate_doc['DocId'], 'Title': duplicate_doc['Title'],
                                         'Abstract': duplicate_doc['Abstract']})
            # print(outliers)
            outliers = sorted(outliers, key=lambda outlier: outlier['DocId'])
            # Save to outlier csv to 'data'
            path = os.path.join('data', 'UrbanStudyCorpus_outliers.csv')
            outlier_df = pd.DataFrame(outliers, columns=['DocId', 'Title', 'Abstract'])
            outlier_df.to_csv(path, encoding='utf-8', index=False)
            # Return a list of doc ids
            return list(map(lambda doc: doc['DocId'], outliers))
        except Exception as err:
            print("Error occurred! {err}".format(err=err))




