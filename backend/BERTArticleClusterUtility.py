import os
import re
import logging
import string
from functools import reduce
from pathlib import Path

import hdbscan
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


# Utility for finding article clusters
class BERTArticleClusterUtility:
    case_name = 'AIMLUrbanStudyCorpus'
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
        _path = os.path.join(BERTArticleClusterUtility.output_path,
                             BERTArticleClusterUtility.case_name + '_clusters.json')
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
                qualified_tags = ['NN', 'NNS', 'JJ', 'NNP']
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
                            _word.lower() in BERTArticleClusterUtility.stop_words or _pos_tag not in qualified_tags:
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
                            score = BERTArticleClusterUtility.compute_Silhouette_score(
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
            _corpus_df = pd.read_csv(os.path.join(_folder, case_name + '.csv'))
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

    # Collect cluster #0 as a new corpus
    @staticmethod
    def collect_cluster_as_corpus(case_name, cluster_no):
        # Load the cluster results
        path = os.path.join('output', case_name, 'iteration', case_name + '_clusters.json')
        corpus = pd.read_json(path).to_dict("records")
        # Get the papers of the cluster 
        cluster_docs = list(filter(lambda doc: doc['Cluster'] == cluster_no, corpus))
        df = pd.DataFrame(cluster_docs)
        # Re_order the columns (Cluster,DocId,Cited by,Year,Document Type,Title,Abstract,Author Keywords,Authors,DOI)
        df = df[['DocId', 'Cited by', 'Year', 'Document Type', 'Title', 'Abstract', 'Author Keywords',
                 'Authors', 'DOI']]
        folder = os.path.join('data', case_name, 'cluster_' + str(cluster_no), "iteration_0")
        Path(folder).mkdir(parents=True, exist_ok=True)
        # Output the cluster data to a corpus
        path = os.path.join(folder, case_name + '_cleaned.csv')
        df.to_csv(path, encoding='utf-8', index=False)
        path = os.path.join(folder, case_name + '_cleaned.json')
        df.to_json(path, orient='records')