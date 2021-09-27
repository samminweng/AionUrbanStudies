import os
from argparse import Namespace
import logging

import hdbscan
import pandas as pd
import numpy as np
import nltk
# # Sentence Transformer (https://www.sbert.net/index.html)
from sentence_transformers import SentenceTransformer
from nltk.tokenize import sent_tokenize, word_tokenize
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
            model_name='all-mpnet-base-v2',
            min_cluster_size=10
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

        # Store the clustering results
        self.result_df = pd.DataFrame(columns=['DocId', 'Text', 'x', 'y'])
        self.result_df['DocId'] = self.documents['DocId']
        self.result_df['Text'] = self.documents['document']
        self.result_df['x'] = list(map(lambda x: round(x, 2), self.clusterable_embedding[:, 0]))
        self.result_df['y'] = list(map(lambda x: round(x, 2), self.clusterable_embedding[:, 1]))

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
            pickle.dump({'documents': documents, 'embeddings': sentences_embeddings}, fOut,
                        protocol=pickle.HIGHEST_PROTOCOL)

    # Get the sentence embedding and cluster doc by hdbscan (https://hdbscan.readthedocs.io/en/latest/index.html)
    def cluster_doc_by_hdbscan(self):
        try:
            # Cluster the documents with minimal cluster size using HDBSCAN
            # Ref: https://hdbscan.readthedocs.io/en/latest/index.html
            clusters = hdbscan.HDBSCAN(min_samples=1, min_cluster_size=self.args.min_cluster_size,  # leaf_size=40,
                                       metric='euclidean').fit(self.clusterable_embedding)

            # clusters.condensed_tree_.plot()
            self.result_df['HDBSCAN_Cluster'] = clusters.labels_
        except Exception as err:
            print("Error occurred! {err}".format(err=err))

    def cluster_doc_by_agglomerative(self):
        # Normalize the embeddings to unit length
        # corpus_embeddings = self.document_embeddings / np.linalg.norm(self.document_embeddings, axis=1, keepdims=True)
        # Perform Agglomerative clustering
        clustering_model = AgglomerativeClustering(n_clusters=None, distance_threshold=1.5)
        clustering_model.fit(self.clusterable_embedding)
        clusters = clustering_model.labels_
        self.result_df['Agglomerative_Cluster'] = clusters

    # Cluster document embedding by KMeans clustering
    def cluster_doc_by_KMeans(self):
        try:
            # sum_of_squared_distances = []  # Hold the SSE value for each K value
            # We use the k-means clustering technique to group 600 documents into 5 groups
            # random_state is the random seed
            num_cluster = 10
            # for num_cluster in range(1, 150):
            clusters = KMeans(n_clusters=num_cluster, random_state=42).fit(self.clusterable_embedding)
            self.result_df['KMeans_Cluster'] = clusters.labels_
            # Re-order
            self.result_df = self.result_df.reindex(
                columns=['KMeans_Cluster', 'HDBSCAN_Cluster', 'Agglomerative_Cluster',
                         'DocId', 'Text', 'x', 'y'])
            # # Write the result to csv and json file
            path = os.path.join(self.output_path, self.args.case_name + '_clusters.csv')
            self.result_df.to_csv(path, encoding='utf-8', index=False)
            # # # Write to a json file
            path = os.path.join(self.output_path, self.args.case_name + '_clusters.json')
            self.result_df.to_json(path, orient='records')
            print('Output cluster results and 2D data points to ' + path)
            # sum_of_squared_distances.append({'cluster': num_cluster, 'sse': clusters.inertia_})
        except Exception as err:
            print("Error occurred! {err}".format(err=err))

    # Derive the topic words from each cluster of documents
    def derive_topic_words_from_cluster_docs(self):
        cluster_approaches = ['KMeans_Cluster', 'HDBSCAN_Cluster', 'Agglomerative_Cluster']
        try:
            # Load the corpus
            corpus_df = pd.read_json(os.path.join('data', self.args.case_name + '.json'))
            # Load the document cluster
            doc_clusters_df = pd.read_json(
                os.path.join(self.output_path, self.args.case_name + '_clusters.json'))
            # Cluster the documents by
            for cluster_approach in cluster_approaches:
                # Group the documents and doc_id by clusters
                docs_per_cluster = doc_clusters_df.groupby([cluster_approach], as_index=False).agg(
                    {'DocId': lambda doc_id: list(doc_id), 'Text': lambda text: list(text)})
                # Derive topic words using C-TF-IDF
                tf_idf, count = TopicUtility.compute_c_tf_idf_score(docs_per_cluster['Text'], len(doc_clusters_df))
                # Top_n_word is a dictionary where key is the cluster no and the value is a list of topic words
                top_n_words = TopicUtility.extract_top_n_words_per_topic(tf_idf, count,
                                                                         docs_per_cluster[cluster_approach],
                                                                         n=50)
                # print(top_n_words)
                results = []
                for i, cluster in docs_per_cluster.iterrows():
                    try:
                        cluster_no = cluster[cluster_approach]
                        doc_ids = cluster['DocId']
                        doc_texts = cluster['Text']
                        # Derive topic words through collocation likelihood
                        topic_words_collocations = TopicUtility.derive_topic_words_using_collocations('likelihood',
                                                                                                      doc_ids, doc_texts)
                        # Derive topic words using C-TF-IDF
                        topic_words_ctf_idf = TopicUtility.group_docs_by_topic_words(doc_ids, doc_texts,
                                                                                     top_n_words[cluster_no])
                        # Derive topic words using TF-IDF
                        topic_words_tf_idf = TopicUtility.derive_topic_words_tf_idf(doc_ids, doc_texts)

                        # # Collect the result
                        result = {"Cluster": cluster_no, 'NumDocs': len(doc_ids), 'DocIds': doc_ids,
                                  'TopicWords_by_Collocations': topic_words_collocations[:50],
                                  'TopicWords_by_CTF-IDF': topic_words_ctf_idf[:50],
                                  'TopicWords_by_TF-IDF': topic_words_tf_idf[:50]
                                  }
                        results.append(result)
                    except Exception as err:
                        print("Error occurred! {err}".format(err=err))
                # Write the result to csv and json file
                cluster_df = pd.DataFrame(results, columns=['Cluster', 'NumDocs', 'DocIds',
                                                            'TopicWords_by_TF-IDF',
                                                            'TopicWords_by_CTF-IDF',
                                                            'TopicWords_by_Collocations'])
                path = os.path.join(self.output_path,
                                    self.args.case_name + '_' + cluster_approach + '_topic_words.csv')
                cluster_df.to_csv(path, encoding='utf-8', index=False)
                # # # Write to a json file
                path = os.path.join(self.output_path,
                                    self.args.case_name + '_' + cluster_approach + '_topic_words.json')
                cluster_df.to_json(path, orient='records')
                print('Output keywords/phrases to ' + path)
        except Exception as err:
            print("Error occurred! {err}".format(err=err))


# Main entry
if __name__ == '__main__':
    docCluster = DocumentCluster()
    # docCluster.get_sentence_embedding()
    # docCluster.cluster_doc_by_hdbscan()
    # docCluster.cluster_doc_by_agglomerative()
    # docCluster.cluster_doc_by_KMeans()
    # TopicUtility.visual_KMean_results()
    # TopicUtility.visualise_cluster_results(docCluster.args.min_cluster_size)
    docCluster.derive_topic_words_from_cluster_docs()
