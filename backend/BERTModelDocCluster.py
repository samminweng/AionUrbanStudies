import os
from argparse import Namespace
import logging
import hdbscan
import pandas as pd
import nltk
# # Sentence Transformer (https://www.sbert.net/index.html)
from sentence_transformers import SentenceTransformer
from nltk.tokenize import sent_tokenize, word_tokenize
import umap  # (UMAP) is a dimension reduction technique https://umap-learn.readthedocs.io/en/latest/
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from pathlib import Path

from sklearn.metrics import adjusted_rand_score, adjusted_mutual_info_score

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
        # Reduce the dimension of doc embeddings to 2D for computing cosine similarity
        self.clusterable_embedding = umap.UMAP(
            n_neighbors=100,
            min_dist=0.0,
            n_components=2,
            random_state=42,
            metric='cosine'
        ).fit_transform(self.doc_vectors)
        # Load the Scopus downloaded file as input file
        path = os.path.join('data', self.args.case_name + '.json')
        text_df = pd.read_json(path)
        # Concatenate 'Title' and 'Abstract' into 'Text
        text_df['Title'] = text_df['Title'] + ". "
        text_df['Text'] = text_df['Title'] + text_df['Abstract']
        # Store the clustering results
        self.cluster_df = pd.DataFrame(columns=['DocId', 'Text', 'x', 'y'])
        self.cluster_df['DocId'] = text_df['DocId']
        self.cluster_df['Text'] = text_df['Text']
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

    # Experiment UMAP dimension reduction and evaluate reduced vectors with clustering evaluation metrics
    def evaluate_UMAP_cluster_quality(self, is_experiment=True):
        def collect_cluster_results(_cluster_labels):
            _cluster_results = []
            for _cluster_no in range(-1, max(_cluster_labels) + 1):
                _count = len(list(filter(lambda label: label == _cluster_no, _cluster_labels)))
                _cluster_results.append({'cluster_no': _cluster_no, 'count': _count})
            _outliers = next(c for c in _cluster_results if c['cluster_no'] == -1)
            # Sort cluster results by count
            _cluster_results = sorted(_cluster_results, key=lambda c: c['cluster_no'], reverse=True)
            return _cluster_results, _outliers

        try:
            # Store experiment results
            results = list()
            # Cluster doc vectors without using UMAP dimension reduction
            baseline_labels = hdbscan.HDBSCAN().fit_predict(self.doc_vectors).tolist()
            baseline_results, baseline_outliers = collect_cluster_results(baseline_labels)
            results.append({'n_neighbors': 'None', 'n_components': 'None', 'min_dist': 'None',
                            'outliers': baseline_outliers['count'], 'total_clusters': len(baseline_results),
                            'adjusted_rand_score': adjusted_rand_score(baseline_labels, baseline_labels),
                            'adjusted_mutual_info_score': adjusted_mutual_info_score(baseline_labels, baseline_labels),
                            'cluster_results': baseline_results
                            })
            BERTModelDocClusterUtility.visualise_cluster_results(baseline_labels, self.doc_vectors, 'No_UMAP')

            best_parameters = [{'n_neighbors': 200, 'n_components': 50, 'min_dist': 0.0},
                               {'n_neighbors': 200, 'n_components': 5, 'min_dist': 0.0},
                               {'n_neighbors': 200, 'n_components': 3, 'min_dist': 0.0}]
            for best_parameter in best_parameters:
                reduced_vectors = umap.UMAP(
                    n_neighbors=best_parameter['n_neighbors'],
                    min_dist=best_parameter['min_dist'],
                    n_components=best_parameter['n_components'],
                    random_state=42,
                    metric='cosine').fit_transform(self.doc_vectors)
                # Cluster reduced vectors
                cluster_labels = hdbscan.HDBSCAN().fit_predict(reduced_vectors).tolist()
                BERTModelDocClusterUtility.visualise_cluster_results(cluster_labels, reduced_vectors,
                                                                     'neighbour_{n}_n_component_{c}'.format(
                                                                         n=best_parameter['n_neighbors'],
                                                                         c=best_parameter['n_components']))

            if is_experiment:
                min_dist = 0.0
                # Experiment UMAP clustering with 'n_component' parameters
                for n_neighbors in [200, 150, 100, 50]:
                    for n_components in [768, 600, 500, 400, 300, 200, 100, 50, 30, 15, 10, 5, 4, 3, 2]:
                        parameter = {'n_components': n_components, 'n_neighbors': n_neighbors, 'min_dist': min_dist}
                        result = {'n_neighbors': str(parameter['n_neighbors']),
                                  'n_components': str(parameter['n_components']),
                                  'min_dist': str(parameter['min_dist']), 'outliers': 'Error',
                                  'total_clusters': 'Error',
                                  'adjusted_rand_score': 'Error', 'adjusted_mutual_info_score': 'Error',
                                  'cluster_results': 'Error'}
                        try:
                            reduced_vectors = umap.UMAP(
                                n_neighbors=parameter['n_neighbors'],
                                min_dist=parameter['min_dist'],
                                n_components=parameter['n_components'],
                                random_state=42,
                                metric='cosine').fit_transform(self.doc_vectors)
                            # Cluster reduced vectors
                            cluster_labels = hdbscan.HDBSCAN().fit_predict(reduced_vectors).tolist()
                            cluster_results, outliers = collect_cluster_results(cluster_labels)
                            # Sort cluster result
                            result['outliers'] = outliers['count']
                            result['total_clusters'] = len(cluster_results)
                            result['cluster_results'] = cluster_results
                            result['adjusted_rand_score'] = adjusted_rand_score(baseline_labels, cluster_labels)
                            result['adjusted_mutual_info_score'] = adjusted_mutual_info_score(baseline_labels,
                                                                                              cluster_labels)
                        except Exception as err:
                            print("Error occurred! {err}".format(err=err))
                        print(result)
                        results.append(result)
                df = pd.DataFrame(results, columns=['n_neighbors', 'n_components', 'total_clusters', 'outliers',
                                                    'cluster_results', 'adjusted_rand_score',
                                                    'adjusted_mutual_info_score'])
                # Output cluster results to CSV
                path = os.path.join('output', 'cluster', 'experiments', 'umap',
                                    'UMAP_HDBSCAN_cluster_doc_vector_results.csv')
                df.to_csv(path, encoding='utf-8', index=False)
        except Exception as err:
            print("Error occurred! {err}".format(err=err))

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
    mdc.evaluate_UMAP_cluster_quality()
    # mdc.cluster_doc_by_hdbscan(is_graph=False)
    # mdc.cluster_doc_by_KMeans()
    # BERTModelDocClusterUtility.visualise_cluster_results_by_methods()
    # BERTModelDocClusterUtility.visualise_cluster_results_by_cluster_number(9)
    # mdc.derive_topics_from_cluster_docs_by_TF_IDF()
    # mdc.combine_topics_from_clusters()
