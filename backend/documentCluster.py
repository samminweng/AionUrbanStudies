import os
from argparse import Namespace
import logging
import pandas as pd
import nltk
# # Sentence Transformer
# # https://www.sbert.net/index.html
from sentence_transformers import SentenceTransformer
from nltk.tokenize import sent_tokenize
import umap     # (UMAP) is a dimension reduction technique https://umap-learn.readthedocs.io/en/latest/
import hdbscan
from sklearn.cluster import KMeans
# import matplotlib.pyplot as plt
# from sklearn.datasets import fetch_20newsgroups
from TopicCreator import TopicCreator

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
            path='data'
        )
        path = os.path.join('data', self.args.case_name + '.csv')
        text_df = pd.read_csv(path)
        self.data = list()
        # Search all the subject words
        for i, text in text_df.iterrows():
            try:
                sentences = sent_tokenize(text['Abstract'])
                sentences = list(filter(lambda s: u"\u00A9" not in s.lower() and 'licensee' not in s, sentences))
                paragraph = text['Title'] + ". " + " ".join(sentences)
                self.data.append(paragraph)
            except Exception as err:
                print("Error occurred! {err}".format(err=err))
        # print(self.data)

    # Sentence transformer is based on transformer model (BERTto compute the vectors for sentences or paragraph (a number of sentences)
    def get_sentence_embedding_cluster_sentence(self):
        try:
            path = os.path.join('/Scratch', 'mweng', 'SentenceTransformer')
            model = SentenceTransformer('distilbert-base-nli-mean-tokens', cache_folder=path)
            embeddings = model.encode(self.data, show_progress_bar=True)
            umap_embeddings = umap.UMAP(n_neighbors=15,
                                        n_components=5,
                                        metric='cosine').fit_transform(embeddings)
            # Cluster the documents by using HDBSCAN
            # cluster = hdbscan.HDBSCAN(min_cluster_size=60, min_samples=1,   # alpha=1.3,
            #                           metric='euclidean',
            #                           # cluster_selection_method='eom'
            #                           ).fit(umap_embeddings)
            # We use the k-means clustering technique to group 600 documents into 5 groups
            cluster = KMeans(n_clusters=5, random_state=0).fit(umap_embeddings)
            # # Prepare data and visualise the result
            # umap_data = umap.UMAP(n_neighbors=15, n_components=2, min_dist=0.0, metric='cosine').fit_transform(embeddings)
            # result = pd.DataFrame(umap_data, columns=['x', 'y'])
            # result['labels'] = cluster.labels_
            docs_df = pd.DataFrame(self.data, columns=["Text"])
            docs_df['Cluster'] = cluster.labels_
            docs_df['DocId'] = range(len(docs_df))
            # Re-order columns
            docs_df = docs_df[['Cluster', 'DocId', 'Text']]
            # Write the result to csv and json file
            path = os.path.join('output', 'cluster', self.args.case_name + '_doc_clusters.csv')
            docs_df.to_csv(path, encoding='utf-8', index=False)
            # # Write to a json file
            path = os.path.join('output', 'cluster', self.args.case_name + '_doc_clusters.json')
            docs_df.to_json(path, orient='records')
            print('Output keywords/phrases to ' + path)
        except Exception as err:
            print("Error occurred! {err}".format(err=err))


    # Derive the topic
    def derive_topic_from_cluster_docs(self):
        # Load the cluster
        path = os.path.join('output', 'cluster', self.args.case_name + '_doc_clusters.json')
        docs_df = pd.read_json(path)
        # Group the documents by topics
        docs_per_cluster = docs_df.groupby(['Cluster'], as_index=False).agg({'Text': ' '.join})
        # compute the tf-idf scores for each cluster
        tf_idf, count = TopicCreator.compute_c_tf_idf_score(docs_per_cluster['Text'].values, len(docs_df))
        # print(tf_idf)
        # 'top_words' is a dictionary
        top_words = TopicCreator.extract_top_n_words_per_topic(tf_idf, count, docs_per_cluster)
        # print(top_words)
        doc_ids_per_cluster = docs_df.groupby(['Cluster'], as_index=False).agg({'DocId': lambda doc_id: list(doc_id)})
        # Combine 'doc_id_per_cluster' and 'top_collocations' into the results
        results = []
        for i, cluster in doc_ids_per_cluster.iterrows():
            top_words_per_cluster = list(map(lambda word: word[0],top_words[i]))
            result = {"Cluster": i, 'NumDocs': len(cluster['DocId']), 'DocIds': cluster['DocId'],
                      "TopWords": top_words_per_cluster}
            results.append(result)
        # Write the result to csv and json file
        cluster_df = pd.DataFrame(results, columns=['Cluster', 'NumDocs', 'DocIds', 'TopWords'])
        path = os.path.join('output', 'cluster', self.args.case_name + '_top_words_clusters.csv')
        cluster_df.to_csv(path, encoding='utf-8', index=False)
        # # Write to a json file
        path = os.path.join('output', 'cluster', self.args.case_name + '_top_words_clusters.json')
        cluster_df.to_json(path, orient='records')
        print('Output keywords/phrases to ' + path)




# Main entry
if __name__ == '__main__':
    docCluster = DocumentCluster()
    # docCluster.get_sentence_embedding_cluster_sentence()
    docCluster.derive_topic_from_cluster_docs()
