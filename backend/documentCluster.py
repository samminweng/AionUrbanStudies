import os
from argparse import Namespace
import logging

import hdbscan
import pandas as pd
import numpy as np
import nltk
# # Sentence Transformer (https://www.sbert.net/index.html)
from sentence_transformers import SentenceTransformer
from nltk.tokenize import sent_tokenize
import umap  # (UMAP) is a dimension reduction technique https://umap-learn.readthedocs.io/en/latest/
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from pathlib import Path
from TopicUtility import TopicUtility
from Utility import Utility
import pickle
from sklearn.cluster import AgglomerativeClustering

logging.basicConfig(level=logging.INFO)
nltk_path = os.path.join('/Scratch', 'mweng', 'nltk_data')
Path(nltk_path).mkdir(parents=True, exist_ok=True)
nltk.data.path.append(nltk_path)
nltk.download('punkt', download_dir=nltk_path)
# Download all the necessary NLTK data
nltk.download('stopwords', download_dir=nltk_path)
# Sentence Transformer
sentence_transformers_path = os.path.join('/Scratch', 'mweng', 'SentenceTransformer')
Path(sentence_transformers_path).mkdir(parents=True, exist_ok=True)

# Cluster the document using BERT model
# Ref: https://towardsdatascience.com/topic-modeling-with-bert-779f7db187e6
class DocumentCluster:
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
        Path(self.output_path).mkdir(parents=True, exist_ok=True)
        # Create the image path for image files
        self.image_path = os.path.join('images', 'cluster')
        Path(self.image_path).mkdir(parents=True, exist_ok=True)
        # Load sentence embeddings
        path = os.path.join(self.output_path, self.args.model_name + '_embeddings.pkl')
        with open(path, "rb") as fIn:
            stored_data = pickle.load(fIn)
            self.documents = pd.DataFrame(stored_data['documents'])
            self.document_embeddings = stored_data['embeddings']

        self.clusterable_embedding = umap.UMAP(
            n_neighbors=100,
            min_dist=0.0,
            n_components=2,
            random_state=42,
            metric='cosine'
        ).fit_transform(self.document_embeddings)
        plt.style.use('bmh')

    # Get the sentence embedding from the transformer model
    # Sentence transformer is based on transformer model (BERTto compute the vectors for sentences or paragraph (a number of sentences)
    def get_sentence_embedding(self):
        # Create the topic path for visualisation
        path = os.path.join('data', self.args.case_name + '.csv')
        text_df = pd.read_csv(path)
        documents = list()
        # Search all the subject words
        for i, text in text_df.iterrows():
            try:
                sentences = sent_tokenize(text['Title'] + ". " + text['Abstract'])
                sentences = Utility.clean_sentence(sentences)
                document = " ".join(sentences)
                documents.append({"DocId": i, "document": document})
            except Exception as err:
                print("Error occurred! {err}".format(err=err))

        document_sentences = list(map(lambda doc: doc['document'], documents))
        # Load Sentence Transformer
        model = SentenceTransformer(self.args.model_name, cache_folder=sentence_transformers_path)
        sentences_embeddings = model.encode(document_sentences, show_progress_bar=True)
        path = os.path.join(self.output_path, self.args.model_name + '_embeddings.pkl')
        # Store sentences & embeddings on disc
        with open(path, "wb") as fOut:
            pickle.dump({'documents': documents, 'embeddings': sentences_embeddings}, fOut, protocol=pickle.HIGHEST_PROTOCOL)

    # Get the sentence embedding and cluster doc by hdbscan (https://hdbscan.readthedocs.io/en/latest/index.html)
    def cluster_doc_by_hdbscan(self):
        try:
            # Store the clustering results
            result_df = pd.DataFrame()
            result_df['DocId'] = self.documents['DocId']
            result_df['x'] = self.clusterable_embedding[:, 0]
            result_df['y'] = self.clusterable_embedding[:, 1]
            # Cluster the documents with minimal cluster size using HDBSCAN
            # Ref: https://hdbscan.readthedocs.io/en/latest/index.html
            min_cluster_size = 5
            min_samples = 3
            # min_sample = 15
            clusters = hdbscan.HDBSCAN(min_samples=min_samples, min_cluster_size=min_cluster_size, leaf_size=40,
                                       metric='euclidean').fit(self.clusterable_embedding)

            # clusters.condensed_tree_.plot()
            result_df['labels'] = clusters.labels_


            return result_df
        except Exception as err:
            print("Error occurred! {err}".format(err=err))


    def cluster_doc_by_agglomerative(self):
        # Normalize the embeddings to unit length
        # corpus_embeddings = self.document_embeddings / np.linalg.norm(self.document_embeddings, axis=1, keepdims=True)
        # Perform Agglomerative clustering
        clustering_model = AgglomerativeClustering(n_clusters=None, distance_threshold=1.5)
        clustering_model.fit(self.clusterable_embedding)
        clusters = clustering_model.labels_
        max_cluster = max(clusters)
        print("Agglomerative cluster produces " + str(max_cluster) + " clusters")
        result_df = pd.DataFrame()
        result_df['x'] = self.clusterable_embedding[:, 0]
        result_df['y'] = self.clusterable_embedding[:, 1]
        result_df['labels'] = clusters
        return result_df


    # Cluster document embedding by KMeans clustering
    def cluster_doc_by_KMeans(self):
        try:
            # Store the clustering results
            result_df = pd.DataFrame()
            result_df['x'] = self.clusterable_embedding[:, 0]
            result_df['y'] = self.clusterable_embedding[:, 1]
            sum_of_squared_distances = []  # Hold the SSE value for each K value
            # We use the k-means clustering technique to group 600 documents into 5 groups
            # random_state is the random seed
            num_cluster = 10
            clusters = KMeans(n_clusters=num_cluster, random_state=42).fit(self.clusterable_embedding)
            result_df['labels'] = clusters.labels_
            sum_of_squared_distances.append({'cluster': num_cluster, 'sse': clusters.inertia_})
            return result_df
            # # Write out the cluster results
            # docs_df = pd.DataFrame(self.documents, columns=["Text"])
            # docs_df['Cluster'] = cluster.labels_
            # docs_df['DocId'] = range(len(docs_df))
            # # Round up data point 'x' and 'y' to 2 decimal
            # docs_df['x'] = result_df['x'].apply(lambda x: round(x, 2))
            # docs_df['y'] = result_df['y'].apply(lambda y: round(y, 2))
            # # Re-order columns
            # docs_df = docs_df[['Cluster', 'DocId', 'Text', 'x', 'y']]
            # # Write the result to csv and json file
            # path = os.path.join(self.folder_path,
            #                     self.args.case_name + '_' + str(num_cluster) + '_' + str(n_neighbour) + '_doc_clusters.csv')
            # docs_df.to_csv(path, encoding='utf-8', index=False)
            # # # Write to a json file
            # path = os.path.join(self.folder_path,
            #                     self.args.case_name + '_' + str(num_cluster) + '_' + str(n_neighbour) + '_doc_clusters.json')
            # docs_df.to_json(path, orient='records')
            # print('Output cluster results and 2D data points to ' + path)
        except Exception as err:
            print("Error occurred! {err}".format(err=err))

    # # Use 'elbow method' to vary cluster number for selecting an optimal K value
    # # The elbow point of the curve is the optimal K value
    def draw_optimal_cluster_for_KMean(self):
        path = os.path.join(self.folder_path, 'k_values', self.args.case_name + '_k_value_cluster.json')
        sse_df = pd.read_json(path)
        clusters = range(1, 100)
        #
        # fig, axs = plt.subplots(nrows=1, ncols=3)
        try:
            for i, n_neighbour in enumerate([15, 50, 100]):
                fig, ax = plt.subplots()
                data_points = sse_df.query('n_neighbour == @n_neighbour')
                sse_values = data_points['sse'].tolist()[:100]
                clusters = data_points['cluster'].tolist()[:100]
                ax.plot(clusters, sse_values)
                ax.set_xlim(0, 100)
                ax.set_ylim(0, 2500)
                ax.set_xticks(np.arange(0, 101, 5))
                ax.set_xlabel('Number of Clusters')
                ax.set_ylabel('Sum of Square Distances')
                ax.set_title('KMean Value Curve (UMAP neighbour = ' + str(n_neighbour) + ")")
                ax.scatter(5, round(sse_values[5]), marker="x")
                ax.scatter(10, round(sse_values[10]), marker="x")
                ax.scatter(15, round(sse_values[15]), marker="x")
                ax.scatter(20, round(sse_values[20]), marker="x")
                # plt.grid(True)
                fig.show()
                path = os.path.join(self.image_path, "elbow_curve", "neighbour_" + str(n_neighbour) + ".png")
                fig.savefig(path)

        except Exception as err:
            print("Error occurred! {err}".format(err=err))

    # Derive the topic words from each cluster of documents
    def derive_topic_words_from_cluster_docs(self):
        try:
            # Go through different cluster number
            for num_cluster in self.args.num_clusters:
                # Load the cluster
                docs_df = pd.read_json(
                    os.path.join(self.folder_path, self.args.case_name + '_' + str(num_cluster) + '_doc_clusters.json'))
                # Group the documents by topics
                doc_ids_per_cluster = docs_df.groupby(['Cluster'], as_index=False).agg(
                    {'DocId': lambda doc_id: list(doc_id)})
                results = []
                for i, cluster in doc_ids_per_cluster.iterrows():
                    cluster_doc_ids = cluster['DocId']
                    # Collect a list of clustered document where each document is a list of tokens
                    cluster_documents = TopicUtility.collect_docs_by_cluster(self.text_df, cluster_doc_ids)
                    # Derive top 10 topic words (collocations) through Chi-square
                    topic_words_chi = TopicUtility.derive_topic_words('chi', cluster_documents)
                    # Derive top 10 topic words through PMI (pointwise mutual information)
                    topic_words_pmi = TopicUtility.derive_topic_words('pmi', cluster_documents)
                    # Derive topic words through likelihood
                    topic_words_likelihood = TopicUtility.derive_topic_words('likelihood', cluster_documents)
                    # Collect the result
                    result = {"Cluster": i, 'NumDocs': len(cluster['DocId']), 'DocIds': cluster['DocId'],
                              "Topic_Words_chi": topic_words_chi, 'Topic_Words_likelihood': topic_words_likelihood,
                              "Topic_Words_pmi": topic_words_pmi}
                    results.append(result)
                # Write the result to csv and json file
                cluster_df = pd.DataFrame(results, columns=['Cluster', 'NumDocs', 'DocIds', 'Topic_Words_chi',
                                                            'Topic_Words_likelihood', 'Topic_Words_pmi'])
                path = os.path.join(self.topic_path,
                                    self.args.case_name + '_' + str(num_cluster) + '_cluster_topic_words.csv')
                cluster_df.to_csv(path, encoding='utf-8', index=False)
                # # Write to a json file
                path = os.path.join(self.topic_path,
                                    self.args.case_name + '_' + str(num_cluster) + '_cluster_topic_words.json')
                cluster_df.to_json(path, orient='records')
                print('Output keywords/phrases to ' + path)
                # Removed the texts from doc_cluster to reduce the file size for better speed
                docs_df.drop('Text', inplace=True, axis=1)  # axis=1 indicates the removal of 'Text' columns.
                # Save the doc cluster to another file
                path = os.path.join(self.topic_path,
                                    self.args.case_name + '_' + str(num_cluster) + '_simplified_cluster_doc.csv')
                docs_df.to_csv(path, encoding='utf-8', index=False)
                # # Write to a json file
                path = os.path.join(self.topic_path,
                                    self.args.case_name + '_' + str(num_cluster) + '_simplified_cluster_doc.json')
                docs_df.to_json(path, orient='records')
        except Exception as err:
            print("Error occurred! {err}".format(err=err))


# Main entry
if __name__ == '__main__':
    docCluster = DocumentCluster()
    docCluster.get_sentence_embedding()
    # hdbscan_result_df = docCluster.cluster_doc_by_hdbscan()
    # kmeans_result_df = docCluster.cluster_doc_by_KMeans()
    # agglomerative_result_df = docCluster.cluster_doc_by_agglomerative()
    # TopicUtility.visualise_cluster_results(hdbscan_result_df, kmeans_result_df, agglomerative_result_df)
    # docCluster.visual_doc_cluster()
    # docCluster.derive_topic_words_from_cluster_docs()
