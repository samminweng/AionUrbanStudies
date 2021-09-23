import os
from argparse import Namespace
import logging
import pandas as pd
import nltk
# # Sentence Transformer (https://www.sbert.net/index.html)
from sentence_transformers import SentenceTransformer
from nltk.tokenize import sent_tokenize
import umap  # (UMAP) is a dimension reduction technique https://umap-learn.readthedocs.io/en/latest/
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from pathlib import Path
from TopicUtility import TopicUtility

logging.basicConfig(level=logging.INFO)
path = os.path.join('/Scratch', 'mweng', 'nltk_data')
Path(path).mkdir(parents=True, exist_ok=True)
nltk.data.path.append(path)
nltk.download('punkt', download_dir=path)
# Download all the necessary NLTK data
nltk.download('stopwords', download_dir=path)


# Cluster the document using BERT model
# Ref: https://towardsdatascience.com/topic-modeling-with-bert-779f7db187e6
class DocumentCluster:
    def __init__(self):
        self.args = Namespace(
            case_name='UrbanStudyCorpus',
            path='data',
            cluster='KMeans',
            num_clusters=[15, 30, 50],
            dimension=384,
            neighbours=[15, 20, 50, 100, 200]
        )
        # Create the folder path for output clustering files (csv and json)
        self.folder_path = os.path.join('output', 'cluster', self.args.cluster + '_d_' + str(self.args.dimension))
        Path(self.folder_path).mkdir(parents=True, exist_ok=True)
        # Create the image path for image files
        self.image_path = os.path.join('images', 'cluster', self.args.cluster + '_d_' + str(self.args.dimension))
        Path(self.image_path).mkdir(parents=True, exist_ok=True)
        # Create the topic path for visualisation
        self.topic_path = os.path.join('output', 'cluster', self.args.cluster + '_d_' + str(self.args.dimension), 'topic')
        Path(self.topic_path).mkdir(parents=True, exist_ok=True)
        path = os.path.join('data', self.args.case_name + '.csv')
        self.text_df = pd.read_csv(path)
        # Save the text_df to JSON file
        self.text_df.to_json(os.path.join('data', self.args.case_name + '.json'), orient='records')
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
    def get_sentence_embedding_cluster_doc_by_KMeans(self):
        try:
            path = os.path.join('/Scratch', 'mweng', 'SentenceTransformer')
            Path(path).mkdir(parents=True, exist_ok=True)
            # We switched to 'sentence-transformers/all-mpnet-base-v2' which is suitable for clustering with
            # 384 dimensional dense vectors
            # https://huggingface.co/sentence-transformers/all-mpnet-base-v2
            model = SentenceTransformer('all-mpnet-base-v2', cache_folder=path)
            doc_embeddings = model.encode(self.documents, show_progress_bar=True)
            for n_neighbour in self.args.neighbours:
                # We increase the neighbors
                umap_embeddings = umap.UMAP(n_neighbors=n_neighbour,
                                            n_components=self.args.dimension,
                                            metric='cosine').fit_transform(doc_embeddings)
                # Map the document embeddings to 2d for visualisation.
                umap_data_points = umap.UMAP(n_neighbors=100, n_components=2, min_dist=0.0, metric='cosine').fit_transform(
                    doc_embeddings)
                result_df = pd.DataFrame(umap_data_points, columns=['x', 'y'])
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
                    path = os.path.join(self.folder_path,
                                        self.args.case_name + '_' + str(num_cluster) + '_' + str(n_neighbour) + '_doc_clusters.csv')
                    docs_df.to_csv(path, encoding='utf-8', index=False)
                    # # Write to a json file
                    path = os.path.join(self.folder_path,
                                        self.args.case_name + '_' + str(num_cluster) + '_' + str(n_neighbour) + '_doc_clusters.json')
                    docs_df.to_json(path, orient='records')
                    print('Output cluster results and 2D data points to ' + path)
        except Exception as err:
            print("Error occurred! {err}".format(err=err))

    # Visualise the data points
    def visual_doc_cluster(self):
        for num_cluster in self.args.num_clusters:
            path = os.path.join(self.folder_path, self.args.case_name + '_' + str(num_cluster) + '_doc_clusters.json')
            doc_cluster_df = pd.read_json(path)
            fig, ax = plt.subplots(figsize=(10, 10))
            clustered = doc_cluster_df.loc[doc_cluster_df['Cluster'] != -1, :]
            plt.scatter(clustered['x'], clustered['y'], c=clustered['Cluster'], s=2.0, cmap='hsv_r')
            plt.colorbar()
            path = os.path.join(self.image_path, self.args.case_name + '_' + str(num_cluster) + "_cluster.png")
            plt.savefig(path)

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
    docCluster.get_sentence_embedding_cluster_doc_by_KMeans()
    # docCluster.visual_doc_cluster()
    # docCluster.derive_topic_words_from_cluster_docs()
