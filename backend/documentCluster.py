import os
from argparse import Namespace
import logging
import pandas as pd
import nltk
# # Sentence Transformer
# # https://www.sbert.net/index.html
from sentence_transformers import SentenceTransformer
from nltk.tokenize import sent_tokenize
import numpy as np
import umap     # (UMAP) is a dimension reduction technique https://umap-learn.readthedocs.io/en/latest/
import hdbscan
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from TopicCreator import TopicCreator
from pathlib import Path

logging.basicConfig(level=logging.INFO)
path = os.path.join('/Scratch', 'mweng', 'anaconda3', 'envs', 'tf_gpu', 'nltk_data')
nltk.download('punkt', download_dir=path)

# Cluster the document using BERT model
# Ref: https://towardsdatascience.com/topic-modeling-with-bert-779f7db187e6
# Ref: https://mccormickml.com/2019/05/14/BERT-word-embeddings-tutorial/
class DocumentCluster:
    def __init__(self):
        self.args = Namespace(
            case_name='UrbanStudyCorpus',
            path='data',
            num_clusters=[5, 10, 15],
            dimension=384
        )
        # Create the folder path for output files (csv and json)
        self.folder_path = os.path.join('output', 'cluster', 'd_' + str(self.args.dimension))
        Path(self.folder_path).mkdir(parents=True, exist_ok=True)
        # Create the image path for image files
        self.image_path = os.path.join('images', 'cluster', 'd_' + str(self.args.dimension))
        Path(self.image_path).mkdir(parents=True, exist_ok=True)
        path = os.path.join('data', self.args.case_name + '.csv')
        self.text_df = pd.read_csv(path)
        self.documents = list()
        # Search all the subject words
        for i, text in self.text_df.iterrows():
            try:
                sentences = sent_tokenize(text['Abstract'])
                sentences = list(filter(lambda s: u"\u00A9" not in s.lower() and 'licensee' not in s, sentences))
                paragraph = text['Title'] + ". " + " ".join(sentences)
                self.documents.append(paragraph)
            except Exception as err:
                print("Error occurred! {err}".format(err=err))
        # print(self.data)

    # Sentence transformer is based on transformer model (BERTto compute the vectors for sentences or paragraph (a number of sentences)
    def get_sentence_embedding_cluster_doc(self):
        try:
            path = os.path.join('/Scratch', 'mweng', 'SentenceTransformer')
            # 'distilbert-base-nli-mean-tokens' is depreciated
            # https://huggingface.co/sentence-transformers/distilbert-base-nli-mean-tokens
            #model = SentenceTransformer('distilbert-base-nli-mean-tokens', cache_folder=path)
            # As such we switched to 'sentence-transformers/all-mpnet-base-v2' which is suitable for clustering with
            # 384 dimensional dense vectors
            # https://huggingface.co/sentence-transformers/all-mpnet-base-v2
            model = SentenceTransformer('all-mpnet-base-v2', cache_folder=path)
            doc_embeddings = model.encode(self.documents, show_progress_bar=True)
            umap_embeddings = umap.UMAP(n_neighbors=15,
                                        n_components=self.args.dimension,
                                        metric='cosine').fit_transform(doc_embeddings)
            # Map the document embeddings to 2d for visualisation.
            umap_data_points = umap.UMAP(n_neighbors=15, n_components=2, min_dist=0.0, metric='cosine').fit_transform(
                doc_embeddings)
            result_df = pd.DataFrame(umap_data_points, columns=['x', 'y'])
            # Cluster the documents by using HDBSCAN
            # cluster = hdbscan.HDBSCAN(min_cluster_size=60, min_samples=1,   # alpha=1.3,
            #                           metric='euclidean',
            #                           # cluster_selection_method='eom'
            #                           ).fit(umap_embeddings)
            # We use the k-means clustering technique to group 600 documents into 5 groups
            # random_state is the random seed
            for num_cluster in self.args.num_clusters:
                cluster = KMeans(n_clusters=num_cluster, random_state=42).fit(umap_embeddings)
                # Write out the cluster results
                docs_df = pd.DataFrame(self.documents, columns=["Text"])
                docs_df['Cluster'] = cluster.labels_
                docs_df['DocId'] = range(len(docs_df))
                # Round up data point 'x' and 'y' to 2 decimal
                docs_df['x'] = result_df['x'].apply(lambda x: round(x, 2))
                docs_df['y'] = result_df['y'].apply(lambda y: round(y, 2))
                # Re-order columns
                docs_df = docs_df[['Cluster', 'DocId', 'Text', 'x', 'y']]

                # Write the result to csv and json file
                path = os.path.join(self.folder_path, self.args.case_name + '_' + str(num_cluster) + '_doc_clusters.csv')
                docs_df.to_csv(path, encoding='utf-8', index=False)
                # # Write to a json file
                path = os.path.join(self.folder_path, self.args.case_name + '_' + str(num_cluster) + '_doc_clusters.json')
                docs_df.to_json(path, orient='records')
                print('Output cluster results and 2D data points to ' + path)
        except Exception as err:
            print("Error occurred! {err}".format(err=err))

    # Visualise the data points
    def visual_doc_cluster(self):
        for num_cluster in self.args.num_clusters:
            path = os.path.join(self.folder_path, self.args.case_name + '_' + str(num_cluster) + '_doc_clusters.json')
            doc_cluster_df = pd.read_json(path)
            # # Get the max and min of 'x' and 'y'
            # max_x = doc_cluster_df['x'].max()
            # max_y = doc_cluster_df['y'].max()
            # min_x = doc_cluster_df['x'].min()
            # min_y = doc_cluster_df['y'].min()
            fig, ax = plt.subplots(figsize=(10, 10))
            clustered = doc_cluster_df.loc[doc_cluster_df['Cluster'] != -1, :]
            plt.scatter(clustered['x'], clustered['y'], c=clustered['Cluster'], s=2.0, cmap='hsv_r')
            plt.colorbar()
            path = os.path.join(self.image_path, self.args.case_name + '_' + str(num_cluster) + "_cluster.png")
            plt.savefig(path)

    # Derive the topic words from each cluster of documents
    def derive_topic_words_from_cluster_docs(self):
        # Go through different cluster number
        for num_cluster in self.args.num_clusters:
            # Load the cluster
            path = os.path.join('output', 'cluster', self.args.case_name + '_' + str(num_cluster) + '_doc_clusters.json')
            docs_df = pd.read_json(path)
            # Group the documents by topics
            docs_per_cluster = docs_df.groupby(['Cluster'], as_index=False).agg({'Text': ' '.join})
            # compute the tf-idf scores for each cluster
            tf_idf, count = TopicCreator.compute_c_tf_idf_score(docs_per_cluster['Text'].values, len(docs_df))
            # 'top_words' is a dictionary
            top_words = TopicCreator.extract_top_n_words_per_topic(tf_idf, count, docs_per_cluster)
            doc_ids_per_cluster = docs_df.groupby(['Cluster'], as_index=False).agg({'DocId': lambda doc_id: list(doc_id)})
            # Combine 'doc_id_per_cluster' and 'top_collocations' into the results
            results = []
            for i, cluster in doc_ids_per_cluster.iterrows():
                top_words_per_cluster = list(map(lambda word: word[0], top_words[i]))
                topic_words = []
                for topic_word in top_words_per_cluster:
                    topic_doc_ids = TopicCreator.get_doc_ids_by_topic_words(self.text_df, cluster['DocId'], topic_word)
                    if len(topic_doc_ids) > 0:
                        topic_words.append({'topic': topic_word, 'doc_ids': topic_doc_ids})
                topic_words = topic_words[:10]      # Get top 10 topics
                topic_words = sorted(topic_words, key=lambda item: len(item['doc_ids']), reverse=True)
                result = {"Cluster": i, 'NumDocs': len(cluster['DocId']), 'DocIds': cluster['DocId'],
                          "TopWords": topic_words}
                results.append(result)
            # Write the result to csv and json file
            cluster_df = pd.DataFrame(results, columns=['Cluster', 'NumDocs', 'DocIds', 'TopWords'])
            path = os.path.join('output', 'cluster', 'result', self.args.case_name + '_' + str(num_cluster) + '_cluster_topic_words.csv')
            cluster_df.to_csv(path, encoding='utf-8', index=False)
            # # Write to a json file
            path = os.path.join('output', 'cluster', 'result', self.args.case_name + '_' + str(num_cluster) + '_cluster_topic_words.json')
            cluster_df.to_json(path, orient='records')
            print('Output keywords/phrases to ' + path)
            # Removed the texts from doc_cluster to reduce the file size for better speed
            docs_df.drop('Text', inplace=True, axis=1)  # axis=1 indicates the removal of 'Text' columns.
            # Save the doc cluster to another file
            path = os.path.join('output', 'cluster', 'result',
                                self.args.case_name + '_' + str(num_cluster) + '_simplified_cluster_doc.csv')
            docs_df.to_csv(path, encoding='utf-8', index=False)
            # # Write to a json file
            path = os.path.join('output', 'cluster', 'result',
                                self.args.case_name + '_' + str(num_cluster) + '_simplified_cluster_doc.json')
            docs_df.to_json(path, orient='records')


# Main entry
if __name__ == '__main__':
    docCluster = DocumentCluster()
    docCluster.get_sentence_embedding_cluster_doc()
    #docCluster.visual_doc_cluster()
    # docCluster.derive_topic_words_from_cluster_docs()
