import os
from argparse import Namespace
import logging
import hdbscan
import numpy as np
import pandas as pd
import nltk
# # Sentence Transformer (https://www.sbert.net/index.html)
import sklearn
from sentence_transformers import SentenceTransformer
from nltk.tokenize import sent_tokenize, word_tokenize
import umap  # (UMAP) is a dimension reduction technique https://umap-learn.readthedocs.io/en/latest/
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from pathlib import Path

from sklearn.metrics import adjusted_rand_score, adjusted_mutual_info_score, pairwise_distances

from BERTModelDocClusterUtility import BERTModelDocClusterUtility
import pickle
import seaborn as sns  # statistical graph library
import getpass

# Set logging level
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
# Set NLTK data path
nltk_path = os.path.join('/Scratch', getpass.getuser(), 'nltk_data')
if os.name == 'nt':
    nltk_path = os.path.join("C:", os.sep, "Users", getpass.getuser(), "nltk_data")
# Download all the necessary NLTK data
nltk.download('punkt', download_dir=nltk_path)
nltk.download('averaged_perceptron_tagger', download_dir=nltk_path)
nltk.download('stopwords', download_dir=nltk_path)
# Append NTLK data path
nltk.data.path.append(nltk_path)
# Set Sentence Transformer path
sentence_transformers_path = os.path.join('/Scratch', getpass.getuser(), 'SentenceTransformer')
if os.name == 'nt':
    sentence_transformers_path = os.path.join("C:", os.sep, "Users", getpass.getuser(), "SentenceTransformer")
Path(sentence_transformers_path).mkdir(parents=True, exist_ok=True)


# Cluster the document using BERT model
# Ref: https://towardsdatascience.com/topic-modeling-with-bert-779f7db187e6
class BERTModelDocCluster:
    def __init__(self):
        self.args = Namespace(
            case_name='UrbanStudyCorpus',
            path='data',
            # model_name='distilbert-base-nli-mean-tokens',
            # We switched to 'sentence-transformers/all-mpnet-base-v2' which is suitable for clustering with
            # 384 dimensional dense vectors (https://huggingface.co/sentence-transformers/all-mpnet-base-v2)
            model_name='all-mpnet-base-v2'
        )
        # Create the folder path for output clustering files (csv and json)
        self.output_path = os.path.join('output', 'cluster')
        # Load doc embeddings of Urban study corpus generated by BERT transformer model
        path = os.path.join(self.output_path, self.args.model_name + '_embeddings.pkl')
        with open(path, "rb") as fIn:
            stored_data = pickle.load(fIn)
            self.doc_vectors = stored_data['embeddings']

        # Load the Scopus downloaded file as input file
        path = os.path.join('data', self.args.case_name + '.json')
        self.text_df = pd.read_json(path)
        # Concatenate 'Title' and 'Abstract' into 'Text
        self.text_df['Text'] = self.text_df['Title'] + ". " + self.text_df['Abstract']
        # Reduce the dimension of doc embeddings to 2D for computing cosine similarity
        self.clusterable_embedding = umap.UMAP(
            n_neighbors=100,
            min_dist=0.0,
            n_components=2,
            random_state=42,
            metric='cosine'
        ).fit_transform(self.doc_vectors)
        #
        # Store the clustering results
        self.cluster_df = pd.DataFrame(columns=['DocId', 'Text', 'x', 'y'])
        self.cluster_df['DocId'] = self.text_df['DocId']
        self.cluster_df['Text'] = self.text_df['Text']
        self.cluster_df['x'] = list(map(lambda x: round(x, 2), self.clusterable_embedding[:, 0]))
        self.cluster_df['y'] = list(map(lambda x: round(x, 2), self.clusterable_embedding[:, 1]))

    # Get the sentence embedding from the transformer model
    # Sentence transformer is based on transformer model (BERTto compute the vectors for sentences or paragraph (a number of sentences)
    def get_sentence_embedding(self):
        def clean_sentence(_sentences):
            # Preprocess the sentence
            cleaned_sentences = list()  # Skip copy right sentence
            for sentence in _sentences:
                if u"\u00A9" not in sentence.lower() and 'licensee' not in sentence.lower() \
                        and 'copyright' not in sentence.lower() and 'rights reserved' not in sentence.lower():
                    try:
                        cleaned_words = word_tokenize(sentence.lower())
                        # Keep alphabetic characters only and remove the punctuation
                        # cleaned_words = list(filter(lambda word: re.match(r'[^\W\d]*$', word), cleaned_words))
                        cleaned_sentences.append(" ".join(cleaned_words))  # merge tokenized words into sentence
                    except Exception as _err:
                        print("Error occurred! {err}".format(err=_err))
            return cleaned_sentences

        # Create the topic path for visualisation
        path = os.path.join('data', self.args.case_name + '.csv')
        text_df = pd.read_csv(path)
        documents = list()
        # Search all the subject words
        for i, text in text_df.iterrows():
            try:
                sentences = sent_tokenize(text['Title'] + ". " + text['Abstract'])
                sentences = clean_sentence(sentences)  # Clean the sentences
                document = " ".join(sentences)
                documents.append({"DocId": text['DocId'], "document": document})
            except Exception as err:
                print("Error occurred! {err}".format(err=err))

        document_sentences = list(map(lambda doc: doc['document'], documents))
        # Load Sentence Transformer
        model = SentenceTransformer(self.args.model_name, cache_folder=sentence_transformers_path)
        sentences_embeddings = model.encode(document_sentences, show_progress_bar=True)
        path = os.path.join(self.output_path, self.args.model_name + '_embeddings.pkl')
        # Store sentences & embeddings on disc
        with open(path, "wb") as fOut:
            pickle.dump({'documents': documents, 'embeddings': sentences_embeddings}, fOut,
                        protocol=pickle.HIGHEST_PROTOCOL)

    # Experiment  dimension reduction and evaluate reduced vectors with clustering evaluation metrics
    def evaluate_HDBSCAN_cluster_quality(self, is_output=False):
        # Collect clustering results and find outliers and the cluster of minimal size
        def collect_cluster_results(_df):
            try:
                _cluster_docs = _df.to_dict("records")
                _min_cluster = []
                _total_clusters = _df['clusters'].max() + 1
                _cluster_results = []
                # Add cluster results
                for _cluster_no in range(-1, _total_clusters):
                    _docs = list(filter(lambda doc: doc['clusters'] == _cluster_no, _cluster_docs))
                    _cluster_results.append({'cluster_no': _cluster_no, 'count': len(_docs)})
                    # if _cluster_no != -1 and _total_clusters == 2 and len(_docs) <= _min_count:
                    #     _min_count = len(_docs)
                    #     _min_cluster = list(map(lambda c: c['DocId'], _docs))
                # Get outliers
                _outliers = next(c for c in _cluster_results if c['cluster_no'] == -1)
                # Sort cluster results by count
                _cluster_results = sorted(_cluster_results, key=lambda c: c['count'])
                return _cluster_results, _outliers
            except Exception as _err:
                print("Error occurred! {err}".format(err=_err))

        try:
            df = pd.DataFrame()
            df['DocId'] = list(range(1, len(self.doc_vectors) + 1))
            df['DocVectors'] = self.doc_vectors.tolist()
            # Load outlier list
            folder = os.path.join('output', 'cluster', 'experiments', 'hdbscan')
            outliers_df = pd.read_csv(os.path.join(folder, 'HDBSCAN_outlier_#0.csv'))
            outlier_doc_ids = outliers_df['DocId'].tolist()
            # Remove all the outliers
            df = df[~df['DocId'].isin(outlier_doc_ids)]
            n_neighbour = 150
            # Reduce the doc vectors to 2 dimension using UMAP dimension reduction for visualisation
            standard_vectors = umap.UMAP(n_neighbors=n_neighbour,
                                         n_components=2,
                                         random_state=42,
                                         metric='cosine').fit_transform(df['DocVectors'].tolist())
            # Experiment HDBSCAN clustering with different dimensions of vectors
            is_graph = False
            for dimension in [None, 500, 300, 200, 100, 50, 30, 15, 10, 5]:
                # Apply UMAP to reduce the dimensions of document vectors
                if dimension:
                    # Run HDBSCAN on reduced dimensional vectors
                    vectors = umap.UMAP(
                        n_neighbors=n_neighbour,
                        min_dist=0.0,
                        n_components=dimension,
                        random_state=42,
                        metric="cosine").fit_transform(df['DocVectors'].tolist())
                else:
                    # Run HDBSCAN on raw vectors
                    dimension = self.doc_vectors.shape[1]
                    vectors = np.vstack(df['DocVectors'])  # Convert to 2D numpy array
                # Store experiment results
                results = list()
                # Experiment HDBSCAN clustering with different parameters
                for min_samples in [None, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]:
                    for min_cluster_size in range(5, 16):
                        for epsilon in [0.0]:
                            parameter = {'dimension': dimension, 'min_cluster_size': min_cluster_size,
                                         'min_samples': min_samples,
                                         'epsilon': epsilon}
                            result = {'dimension': dimension,
                                      'min_cluster_size': str(parameter['min_cluster_size']),
                                      'min_samples': str(parameter['min_samples']),
                                      'epsilon': str(parameter['epsilon']),
                                      'outliers': 'Error',
                                      'total_clusters': 'Error',
                                      'cluster_results': 'Error',
                                      'Silhouette_score': 'Error'}
                            try:
                                # Compute the cosine distance/similarity for each doc vectors
                                distances = pairwise_distances(vectors, metric='cosine')
                                # Cluster reduced vectors
                                cluster_labels = hdbscan.HDBSCAN(min_cluster_size=min_cluster_size,
                                                                 min_samples=min_samples,
                                                                 cluster_selection_epsilon=epsilon,
                                                                 metric='precomputed').fit_predict(
                                    distances.astype('float64')).tolist()
                                df['clusters'] = cluster_labels
                                df['vectors'] = vectors.tolist()
                                cluster_results, outliers = collect_cluster_results(df)
                                score = BERTModelDocClusterUtility.compute_Silhouette_score(df)
                                # Sort cluster result
                                result['outliers'] = outliers['count']
                                result['total_clusters'] = len(cluster_results)
                                result['cluster_results'] = cluster_results
                                result['Silhouette_score'] = score
                                # Output cluster results to png files
                                BERTModelDocClusterUtility.visualise_cluster_results(cluster_labels,
                                                                                     standard_vectors,
                                                                                     parameter,
                                                                                     is_graph=is_graph
                                                                                     )
                            except Exception as err:
                                print("Error occurred! {err}".format(err=err))
                            print(result)
                            results.append(result)
                # Output the clustering results of a dimension
                if not is_graph:
                    # Output the detailed clustering results
                    result_df = pd.DataFrame(results,
                                             columns=['dimension', 'min_samples', 'min_cluster_size', 'epsilon',
                                                      'total_clusters', 'outliers', 'Silhouette_score',
                                                      'cluster_results'])
                    # Output cluster results to CSV
                    path = os.path.join(folder, 'HDBSCAN_cluster_doc_vector_results_' + str(dimension) + '.csv')
                    result_df.to_csv(path, encoding='utf-8', index=False)
                    path = os.path.join(folder, 'HDBSCAN_cluster_doc_vector_results_' + str(dimension) + '.json')
                    result_df.to_json(path, orient='records')
        except Exception as err:
            print("Error occurred! {err}".format(err=err))

            # Get the doc id common to all min_clusters

    def write_summary(self):
        best_results = list()
        folder = os.path.join('output', 'cluster', 'experiments', 'hdbscan')
        for dimension in [768, 500, 300, 200, 100, 50, 30, 15, 10, 5]:
            path = os.path.join(folder, 'HDBSCAN_cluster_doc_vector_results_' + str(dimension) + '.json')
            df = pd.read_json(path)
            results = df.to_dict("records")
            best_result = {'dimension': dimension, 'score': 0}
            for result in results:
                score = result['Silhouette_score']
                # Check if the score is better than 'best' parameter
                if score != 'None' and float(score) > best_result['score']:
                    best_result['Silhouette_score'] = float(score)
                    best_result['min_samples'] = result['min_samples']
                    best_result['min_cluster_size'] = result['min_cluster_size']
                    best_result['epsilon'] = result['epsilon']
                    best_result['total_clusters'] = result['total_clusters']
                    best_result['outliers'] = result['outliers']
            best_results.append(best_result)
        # Output the best clustering results
        best_result_df = pd.DataFrame(best_results,
                                      columns=['dimension', 'min_samples', 'min_cluster_size', 'epsilon',
                                               'Silhouette_score', 'total_clusters', 'outliers',
                                               'cluster_results'])
        path = os.path.join(folder, 'HDBSCAN_cluster_doc_vector_result_summary.csv')
        best_result_df.to_csv(path, encoding='utf-8', index=False)

    # Get the sentence embedding and cluster doc by HDBSCAN (https://hdbscan.readthedocs.io/en/latest/index.html)
    def cluster_doc_by_hdbscan(self, is_graph=False, min_cluster_size=10, min_samples=1,
                               cluster_selection_epsilon=0.2,
                               cluster_selection_method='eom'):
        try:
            # Cluster the documents with minimal cluster size using HDBSCAN
            # Ref: https://hdbscan.readthedocs.io/en/latest/index.html
            clusters = hdbscan.HDBSCAN(min_cluster_size=min_cluster_size,
                                       min_samples=min_samples,
                                       cluster_selection_epsilon=cluster_selection_epsilon,
                                       metric='euclidean',
                                       cluster_selection_method=cluster_selection_method
                                       ).fit(self.clusterable_embedding)
            # Update the cluster to 'cluster_df'
            self.cluster_df['HDBSCAN_Cluster'] = clusters.labels_
            cluster_number = max(self.cluster_df['HDBSCAN_Cluster'])
            outliers = self.cluster_df.loc[self.cluster_df['HDBSCAN_Cluster'] == -1, :]
            parameters = {"min_cluster_size": min_cluster_size,
                          "min_samples": min_samples,
                          "cluster_selection_method": cluster_selection_method,
                          "cluster_selection_epsilon": cluster_selection_epsilon,
                          "cluster_num": cluster_number,
                          "outliers": len(outliers)
                          }
            print(parameters)
            # Save HDBSCAN cluster to 'temp' folder
            path = os.path.join(self.output_path, 'temp', 'HDBSCAN_cluster_num_' + str(cluster_number) + '.csv')
            self.cluster_df.to_csv(path, encoding='utf-8', index=False)
            if is_graph:
                # Output condense tree
                condense_tree = clusters.condensed_tree_
                # Save condense tree to csv
                tree_df = condense_tree.to_pandas()
                path = os.path.join(self.output_path, 'temp', self.args.case_name + '_clusters_tree.csv')
                tree_df.to_csv(path, encoding='utf-8')
                # Plot condense tree graph
                condense_tree.plot(select_clusters=True,
                                   selection_palette=sns.color_palette('deep', 40),
                                   label_clusters=True,
                                   max_rectangles_per_icicle=150)
                image_path = os.path.join('images', 'HDBSCAN' +
                                          '_min_cluster_size_' + str(min_cluster_size) +
                                          '_cluster_num_' + str(cluster_number) +
                                          '_outlier_' + str(len(outliers)) +
                                          '.png')
                plt.savefig(image_path)
                print("Output HDBSCAN clustering image to " + image_path)
            return parameters
        except Exception as err:
            print("Error occurred! {err}".format(err=err))

    # Cluster document embedding by KMeans clustering
    def cluster_doc_by_KMeans(self, num_cluster=9):
        try:
            # Hold the SSE value for each K value
            # We use the k-means clustering technique to group 600 documents into 5 groups
            # random_state is the random seed
            # for num_cluster in range(1, 150):
            clusters = KMeans(n_clusters=num_cluster, random_state=42).fit(self.clusterable_embedding)
            self.cluster_df['KMeans_Cluster'] = clusters.labels_
            # Re-index and re-order the columns of cluster data frame
            re_order_cluster_df = self.cluster_df.reindex(
                columns=['KMeans_Cluster', 'HDBSCAN_Cluster', 'DocId', 'x', 'y'])
            # # Write the result to csv and json file
            path = os.path.join(self.output_path, self.args.case_name + '_clusters.csv')
            re_order_cluster_df.to_csv(path, encoding='utf-8', index=False)
            # # # Write to a json file
            path = os.path.join(self.output_path, self.args.case_name + '_clusters.json')
            re_order_cluster_df.to_json(path, orient='records')
            print('Output cluster results and 2D data points to ' + path)
            # sum_of_squared_distances.append({'cluster': num_cluster, 'sse': clusters.inertia_})
        except Exception as err:
            print("Error occurred! {err}".format(err=err))

    # Derive the topic words from each cluster of documents
    def derive_topics_from_cluster_docs_by_TF_IDF(self):
        cluster_approaches = ['HDBSCAN_Cluster']
        try:
            # Get the duplicate articles.
            # Note the original Scopus file contain duplicated articles (titles are the same)
            duplicate_doc_ids = BERTModelDocClusterUtility.scan_duplicate_articles()
            print("Duplicated articles in " + self.args.case_name + ":")
            print(*duplicate_doc_ids, sep=", ")
            # Load the document cluster
            doc_clusters_df = pd.read_json(
                os.path.join(self.output_path, self.args.case_name + '_clusters.json'))
            # Update text column
            doc_clusters_df['Text'] = self.cluster_df['Text']
            # Drop the documents that are not in the list of duplicated articles
            doc_clusters_df = doc_clusters_df[~doc_clusters_df['DocId'].isin(duplicate_doc_ids)]
            # Cluster the documents by
            for approach in cluster_approaches:
                # Group the documents and doc_id by clusters
                docs_per_cluster = doc_clusters_df.groupby([approach], as_index=False) \
                    .agg({'DocId': lambda doc_id: list(doc_id), 'Text': lambda text: list(text)})
                # Get top 100 topics (1, 2, 3 grams) for each cluster
                n_gram_topic_list = BERTModelDocClusterUtility.get_n_gram_topics(approach, docs_per_cluster)
                # print(topic_words_df)
                results = []
                for i, cluster in docs_per_cluster.iterrows():
                    try:
                        cluster_no = cluster[approach]
                        doc_ids = cluster['DocId']
                        doc_texts = cluster['Text']
                        result = {"Cluster": cluster_no, 'NumDocs': len(doc_ids), 'DocIds': doc_ids}
                        n_gram_topics = []
                        # Collect the topics of 1 gram, 2 gram and 3 gram
                        for n_gram_num in [1, 2, 3]:
                            n_gram_topic = next(n_gram_topic for n_gram_topic in n_gram_topic_list
                                                if n_gram_topic['n_gram'] == n_gram_num)
                            # Collect top 300 topics of a cluster
                            cluster_topics = n_gram_topic['topics'][str(cluster_no)][:300]
                            # Create a mapping between the topic and its associated articles (doc)
                            doc_per_topic = BERTModelDocClusterUtility.group_docs_by_topics(n_gram_num,
                                                                                            doc_ids, doc_texts,
                                                                                            cluster_topics)
                            n_gram_type = 'Topic-' + str(n_gram_num) + '-gram'
                            result[n_gram_type] = doc_per_topic
                            n_gram_topics += doc_per_topic
                        if cluster_no == 2:  # Debugging only
                            print("Cluster 2")
                        result['Topic-N-gram'] = BERTModelDocClusterUtility.merge_n_gram_topic(n_gram_topics)
                        results.append(result)
                    except Exception as err:
                        print("Error occurred! {err}".format(err=err))
                # Write the result to csv and json file
                cluster_df = pd.DataFrame(results, columns=['Cluster', 'NumDocs', 'DocIds',
                                                            'Topic-1-gram',
                                                            'Topic-2-gram',
                                                            'Topic-3-gram',
                                                            'Topic-N-gram'])
                _path = os.path.join(self.output_path, 'topics',
                                     self.args.case_name + '_' + approach + '_TF-IDF_topic_words.csv')
                cluster_df.to_csv(_path, encoding='utf-8', index=False)
                # # # Write to a json file
                _path = os.path.join(self.output_path, 'topics',
                                     self.args.case_name + '_' + approach + '_TF-IDF_topic_words.json')
                cluster_df.to_json(_path, orient='records')
                print('Output topics per cluster to ' + _path)
            # # Output top 50 topics by 1, 2 and 3-grams
            BERTModelDocClusterUtility.flatten_tf_idf_topics(5)  # topics in Cluster 5 about 'temperature' 'urban heat'
            BERTModelDocClusterUtility.flatten_tf_idf_topics(7)  # topics in Cluster 7 about 'Iot' 'traffic'
            BERTModelDocClusterUtility.flatten_tf_idf_topics(9)  # topics in Cluster 9 about 'human mobility'
        except Exception as err:
            print("Error occurred! {err}".format(err=err))

    # Combine TF-IDF and BERT key phrase extraction Topics into a single file
    def combine_topics_from_clusters(self):
        cluster_approach = 'HDBSCAN_Cluster'
        try:
            _path = os.path.join(self.output_path, 'topics',
                                 self.args.case_name + '_' + cluster_approach + '_TF-IDF_topic_words.json')
            tf_idf_df = pd.read_json(_path)
            tf_ids_dict = tf_idf_df.to_dict("records")
            results = list()
            for topic in tf_ids_dict:
                result = {
                    'Cluster': topic['Cluster'],
                    'NumDocs': topic['NumDocs'],
                    'DocIds': topic['DocIds'],
                    'TF-IDF-Topics': topic['Topic-N-gram']
                }
                results.append(result)
            # Write out to csv and json file
            cluster_df = pd.DataFrame(results, columns=['Cluster', 'NumDocs', 'DocIds',
                                                        'TF-IDF-Topics'])
            _path = os.path.join(self.output_path,
                                 self.args.case_name + '_' + cluster_approach + '_TF-IDF_topic_words.csv')
            cluster_df.to_csv(_path, encoding='utf-8', index=False)
            # # # Write to a json file
            _path = os.path.join(self.output_path,
                                 self.args.case_name + '_' + cluster_approach + '_TF-IDF_topic_words.json')
            cluster_df.to_json(_path, orient='records')
            print('Output topics per cluster to ' + _path)
        except Exception as err:
            print("Error occurred! {err}".format(err=err))


# Main entry
if __name__ == '__main__':
    mdc = BERTModelDocCluster()
    # mdc.get_sentence_embedding()
    # mdc.evaluate_HDBSCAN_cluster_quality()
    mdc.write_summary()
    # mdc.cluster_doc_by_hdbscan(is_graph=False)
    # mdc.cluster_doc_by_KMeans()
    # BERTModelDocClusterUtility.visualise_cluster_results_by_methods()
    # BERTModelDocClusterUtility.visualise_cluster_results_by_cluster_number(9)
    # mdc.derive_topics_from_cluster_docs_by_TF_IDF()
    # mdc.combine_topics_from_clusters()
