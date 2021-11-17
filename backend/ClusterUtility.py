import math
import os
import re
import logging
import nltk
import numpy as np
from matplotlib import pyplot as plt
from nltk import BigramCollocationFinder
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from nltk.util import ngrams
from nltk.tokenize import sent_tokenize
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import getpass
import pandas as pd
# Set logging level
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
# Set NLTK data path
nltk_path = os.path.join('/Scratch', getpass.getuser(), 'nltk_data')
if os.name == 'nt':
    nltk_path = os.path.join("C:", os.sep, "Users", getpass.getuser(), "nltk_data")
# Append NTLK data path
nltk.data.path.append(nltk_path)
nltk.download('punkt', download_dir=nltk_path)
# Download all the necessary NLTK data
nltk.download('stopwords', download_dir=nltk_path)

# Utility for deriving the topics from each cluster of documents.
class ClusterUtility:
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
            path = os.path.join(ClusterUtility.image_path, "elbow_curve.png")
            fig.savefig(path)
        except Exception as err:
            print("Error occurred! {err}".format(err=err))

    @staticmethod
    def visualise_cluster_results():
        plt.style.use('bmh')  # Use black white background
        _path = os.path.join(ClusterUtility.output_path, ClusterUtility.case_name + '_clusters.json')
        cluster_result_df = pd.read_json(_path)
        fig, (ax0, ax1, ax2) = plt.subplots(1, 3)
        # # Visualise KMeans
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
        _path = os.path.join(ClusterUtility.image_path, "cluster_" + str(max_cluster_number) + "_outlier_"
                             + str(len(outliers)) + ".png")
        fig.set_size_inches(10, 5)
        fig.savefig(_path, dpi=600)
        print("Output image to " + _path)

    @staticmethod
    def derive_bag_of_words(doc_ids, doc_texts):
        try:
            vec = CountVectorizer(stop_words=ClusterUtility.stop_words)
            bag_of_words = vec.fit_transform(doc_texts)  # Return a bag of words
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
            finder.apply_ngram_filter(lambda w1, w2: w1.lower() in ClusterUtility.function_words or
                                                     w2.lower() in ClusterUtility.function_words)
            finder.apply_ngram_filter(lambda w1, w2: w1.lower() in ClusterUtility.stop_words or
                                                     w2.lower() in ClusterUtility.stop_words)
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
                                if word in ClusterUtility.lemma_nouns:
                                    singular_word = ClusterUtility.lemma_nouns[word]
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
    def get_n_gram_topics(approach, docs_per_cluster, is_load=False):
        if is_load:
            n_gram_topics_df = pd.read_json(os.path.join('output', 'cluster',
                                                         'temp',
                                                         'UrbanStudyCorpus_' + approach + '_n_topics.json'))
            return n_gram_topics_df.to_dict("records")

        # Convert the texts of all clusters into a list of document (a list of sentences) for deriving n-grams
        def _collect_cluster_docs(_docs_per_cluster):
            # Get the clustered texts
            clusters = _docs_per_cluster[approach]
            doc_texts_per_cluster = docs_per_cluster['Text']
            _docs = []
            for i, doc_texts in doc_texts_per_cluster.items():
                doc_id = clusters[i]  # doc id is cluster id
                doc = []
                for doc_text in doc_texts:
                    text = ClusterUtility.preprocess_text(doc_text.strip())
                    sentences = sent_tokenize(text)
                    doc.extend(sentences)
                _docs.append({'cluster': doc_id, 'doc': doc})  # doc: a list of sentences
            # Convert the frequency matrix to data frame
            df = pd.DataFrame(_docs, columns=['cluster', 'doc'])
            # Write to temp output for validation
            df.to_csv(os.path.join('output', 'cluster', 'temp', approach, 'Step_1_UrbanStudyCorpus_cluster_doc.csv'),
                      encoding='utf-8', index=False)
            df.to_json(os.path.join('output', 'cluster', 'temp', approach, 'Step_1_UrbanStudyCorpus_cluster_doc.json'),
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
                        if word.lower() in ClusterUtility.stop_words or \
                                re.search('\d|[^\w]', word.lower()):
                            is_qualified = False
                            break
                    if is_qualified:
                        r_list.append(" ".join(_n_gram))  # Convert n_gram (a list of words) to a string
                return r_list

            # Vectorized the clustered doc text and Keep the Word case unchanged
            frequency_matrix = []
            for doc in docs:
                doc_id = doc['cluster']  # doc id is the cluster no
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
                frequency_matrix.append({'cluster': doc_id, 'freq_table': freq_table})
            # Convert the frequency matrix to data frame
            df = pd.DataFrame(frequency_matrix, columns=['cluster', 'freq_table'])
            # Write to temp output for validation
            df.to_csv(os.path.join('output', 'cluster', 'temp', approach,
                                   'Step_2_UrbanStudyCorpus_frequency_matrix.csv'),
                      encoding='utf-8', index=False)
            df.to_json(os.path.join('output', 'cluster', 'temp', approach,
                                    'Step_2_UrbanStudyCorpus_frequency_matrix.json'),
                       orient='records')
            print('Output topics per cluster to ' + os.path.join('output', 'cluster', 'temp',
                                                                 'Step2_UrbanStudyCorpus_frequency_matrix.json'))
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
                df.to_csv(os.path.join('output', 'cluster', 'temp', approach,
                                       'Step_3_UrbanStudyCorpus_word_doc_table.csv'),
                          encoding='utf-8', index=False)
                df.to_json(os.path.join('output', 'cluster', 'temp', approach,
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
        # Write the results to
        topic_words_df.to_csv(
            os.path.join('output', 'cluster', 'temp', 'UrbanStudyCorpus_' + approach + '_n_topics.csv'),
            encoding='utf-8', index=False)
        # # # Write to a json file
        topic_words_df.to_json(
            os.path.join('output', 'cluster', 'temp', 'UrbanStudyCorpus_' + approach + '_n_topics.json'),
            orient='records')
        return topics_list  # Return a list of dicts

    # Output the cluster topics as a csv file
    @staticmethod
    def flatten_topics(approach, cluster_no):
        try:
            cluster_df = pd.read_json(
                os.path.join('output', 'cluster', 'UrbanStudyCorpus_' + approach + '_Cluster_topic_words.json'))
            clusters = cluster_df.to_dict("records")
            cluster = next(cluster for cluster in clusters if cluster['Cluster'] == cluster_no)
            records = []
            for i in range(50):
                record = {'1-gram': "", '1-gram-score': 0, '1-gram-freq': 0, '1-gram-docs': 0, '1-gram-clusters': 0,
                          '2-gram': "", '2-gram-score': 0, '2-gram-freq': 0, '2-gram-docs': 0, '2-gram-clusters': 0,
                          '3-gram': "", '3-gram-score': 0, '3-gram-freq': 0, '3-gram-docs': 0, '3-gram-clusters': 0,
                          'N-gram': "", 'N-gram-score': 0, 'N-gram-freq': 0, 'N-gram-docs': 0, 'N-gram-clusters': 0,
                          }
                for n_gram_num in ['1-gram', '2-gram', '3-gram', 'N-gram']:
                    if i < len(cluster['Topic' + n_gram_num]):
                        n_gram = cluster['Topic' + n_gram_num][i]
                        record[n_gram_num] = n_gram['topic']
                        record[n_gram_num + '-score'] = n_gram['score']
                        record[n_gram_num + '-freq'] = n_gram['freq']
                        record[n_gram_num + '-docs'] = len(n_gram['doc_ids'])
                        record[n_gram_num + '-clusters'] = len(n_gram['cluster_ids'])
                records.append(record)
            n_gram_df = pd.DataFrame(records)
            _path = os.path.join('output', 'cluster', 'topics',
                                 'UrbanStudyCorpus_HDBSCAN_Cluster_' + str(cluster_no) + '_topics.csv')
            n_gram_df.to_csv(_path, encoding='utf-8', index=False)
            print('Output topics per cluster to ' + _path)
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
            for plural, singular in ClusterUtility.lemma_nouns.items():
                if singular == last_word:
                    plural_word = plural
                    break
            plural_topic = words[:-1] + [plural_word]
            return " ".join(plural_topic)

        try:
            docs_per_topic = []
            # Go through each article and find if each topic appear in the article
            for doc_id, doc_text in zip(doc_ids, doc_texts):
                # Convert the preprocessed text to n_grams
                tokenizes = word_tokenize(ClusterUtility.preprocess_text(doc_text))
                # Obtain the n-grams from the text
                n_grams = list(ngrams(tokenizes, n_gram_num))
                n_grams = list(map(lambda n_gram: " ".join(n_gram), n_grams))
                # For each topic, find out the document ids that contain the topic
                for item in topics_per_cluster:
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
    def scan_duplicate_articles():
        try:
            corpus_df = pd.read_csv(os.path.join('data', 'UrbanStudyCorpus.csv'))
            corpus = corpus_df.to_dict("records")
            duplicate_doc_ids = set()
            for article in corpus:
                doc_id = article['DocId']
                title = article['Title']
                # Find if other article has the same title and author names
                same_articles = list(filter(lambda a: a['Title'].lower().strip() == title.lower().strip() and
                                                      a['DocId'] > doc_id, corpus))
                if len(same_articles):
                    for sa in same_articles:
                        duplicate_doc_ids.add(sa['DocId'])
            return list(duplicate_doc_ids)
        except Exception as err:
            print("Error occurred! {err}".format(err=err))
