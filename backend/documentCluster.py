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
# nltk_path = os.path.join('/Scratch', 'mweng', 'nltk_data')
# Windows path
nltk_path = os.path.join("C:", os.sep, "Users", "sam", "nltk_data")
Path(nltk_path).mkdir(parents=True, exist_ok=True)
nltk.data.path.append(nltk_path)
nltk.download('punkt', download_dir=nltk_path)
# Download all the necessary NLTK data
nltk.download('stopwords', download_dir=nltk_path)
# Sentence Transformer
# sentence_transformers_path = os.path.join('/Scratch', 'mweng', 'SentenceTransformer')
sentence_transformers_path = os.path.join("C:", os.sep, "Users", "sam", "SentenceTransformer")
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
            self.document_embeddings = stored_data['embeddings']

        self.clusterable_embedding = umap.UMAP(
            n_neighbors=100,
            min_dist=0.0,
            n_components=2,
            random_state=42,
            metric='cosine'
        ).fit_transform(self.document_embeddings)
        plt.style.use('bmh')

        # Concatenate 'Title' and 'Abstract' into 'Text
        path = os.path.join('data', self.args.case_name + '.json')
        text_df = pd.read_json(path)
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
        # Create the topic path for visualisation
        path = os.path.join('data', self.args.case_name + '.csv')
        text_df = pd.read_csv(path)
        documents = list()
        # Search all the subject words
        for i, text in text_df.iterrows():
            try:
                sentences = sent_tokenize(text['Title'] + ". " + text['Abstract'])
                sentences = Utility.clean_sentence(sentences)  # Clean the sentences
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

    # Get the sentence embedding and cluster doc by hdbscan (https://hdbscan.readthedocs.io/en/latest/index.html)
    def cluster_doc_by_hdbscan(self):
        try:
            # Cluster the documents with minimal cluster size using HDBSCAN
            # Ref: https://hdbscan.readthedocs.io/en/latest/index.html
            clusters = hdbscan.HDBSCAN(min_samples=1, min_cluster_size=self.args.min_cluster_size,  # leaf_size=40,
                                       metric='euclidean').fit(self.clusterable_embedding)

            # clusters.condensed_tree_.plot()
            self.cluster_df['HDBSCAN_Cluster'] = clusters.labels_
        except Exception as err:
            print("Error occurred! {err}".format(err=err))

    # Cluster document embedding by KMeans clustering
    def cluster_doc_by_KMeans(self):
        try:
            # sum_of_squared_distances = []  # Hold the SSE value for each K value
            # We use the k-means clustering technique to group 600 documents into 5 groups
            # random_state is the random seed
            num_cluster = 15
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

    # Collect the tf-idf terms for each article in a cluster
    # Note the TF-IDF terms are computed based on all the articles in a cluster, not the entire corpus
    def collect_tf_idf_terms_by_cluster(self):
        # Read corpus df
        corpus_df = pd.read_json(os.path.join('data', self.args.case_name + ".json"))
        cluster_approaches = ['KMeans_Cluster', 'HDBSCAN_Cluster']
        try:
            path = os.path.join(self.output_path, self.args.case_name + '_clusters.json')
            # Load the document cluster
            doc_clusters_df = pd.read_json(path)
            # Update text column
            doc_clusters_df['Text'] = self.cluster_df['Text']
            # Cluster the texts by cluster id
            for cluster_approach in cluster_approaches:
                # Group the documents and doc_id by clusters
                docs_per_cluster = doc_clusters_df.groupby([cluster_approach], as_index=False).agg(
                    {'DocId': lambda doc_id: list(doc_id), 'Text': lambda text: list(text)})
                results = []
                for i, cluster in docs_per_cluster.iterrows():
                    try:
                        doc_ids = cluster['DocId']
                        doc_text = cluster['Text']
                        key_terms = TopicUtility.extract_terms_by_TFIDF(doc_ids, doc_text)
                        results.extend(key_terms)
                    except Exception as err:
                        print("Error occurred! {err}".format(err=err))
                # Sort the results by doc_id
                sorted_results = sorted(results, key=lambda item: item['doc_id'])
                cluster_key_terms = list(map(lambda r: r['key_terms'], sorted_results))
                corpus_df[cluster_approach + "_KeyTerms"] = cluster_key_terms
            # Write the result to csv and json file
            update_corpus_df = corpus_df.reindex(columns=['DocId', 'Year', 'Title', 'Abstract', 'Author Keywords',
                                                          'Authors', 'Cited by', 'DOI', 'Link',
                                                          'KMeans_Cluster_KeyTerms',
                                                          'HDBSCAN_Cluster_KeyTerms'])
            path = os.path.join(self.output_path, self.args.case_name + '_doc_terms.csv')
            update_corpus_df.to_csv(path, encoding='utf-8', index=False)
            # # # Write to a json file
            path = os.path.join(self.output_path, self.args.case_name + '_doc_terms.json')
            update_corpus_df.to_json(path, orient='records')
            print('Output keywords/phrases to ' + path)

        except Exception as err:
            print("Error occurred! {err}".format(err=err))

    # Derive the topic words from each cluster of documents
    def derive_topic_words_from_cluster_docs(self):
        # cluster_approaches = ['KMeans_Cluster', 'HDBSCAN_Cluster']
        cluster_approaches = ['KMeans_Cluster']
        try:
            # Load the document cluster
            doc_clusters_df = pd.read_json(
                os.path.join(self.output_path, self.args.case_name + '_clusters.json'))
            total = len(doc_clusters_df)  # Total number of articles
            # Cluster the documents by
            for approach in cluster_approaches:
                # Group the documents and doc_id by clusters
                docs_per_cluster = doc_clusters_df.groupby([approach], as_index=False) \
                    .agg({'DocId': lambda doc_id: list(doc_id), 'Text': lambda text: list(text)})
                # Get top 100 topics (1, 2, 3 grams) for each cluster
                topic_words_df = TopicUtility.get_n_gram_topics(approach, docs_per_cluster, total)
                # print(topic_words_df)
                results = []
                for i, cluster in docs_per_cluster.iterrows():
                    try:
                        cluster_no = cluster[approach]
                        doc_ids = cluster['DocId']
                        doc_texts = cluster['Text']
                        result = {"Cluster": cluster_no, 'NumDocs': len(doc_ids), 'DocIds': doc_ids}
                        n_gram_topics = {}
                        # Collect the topics of 1 gram, 2 gram and 3 gram
                        for n_gram in [1, 2, 3]:
                            n_gram_row = topic_words_df[topic_words_df['n_gram'] == n_gram].iloc[0]
                            cluster_topics = n_gram_row['topics'][str(cluster_no)]
                            # Derive topic words using BERTopic
                            topic_words_bert_topic = TopicUtility.group_docs_by_topics(n_gram, doc_ids, doc_texts,
                                                                                       cluster_topics)
                            n_gram_type = 'Topic' + str(n_gram) + '-gram'
                            result[n_gram_type] = topic_words_bert_topic
                            n_gram_topics[n_gram_type] = topic_words_bert_topic
                        result['TopicN-gram'] = TopicUtility.merge_and_rank_n_gram_topic(n_gram_topics)
                        results.append(result)
                    except Exception as err:
                        print("Error occurred! {err}".format(err=err))
                # Write the result to csv and json file
                cluster_df = pd.DataFrame(results, columns=['Cluster', 'NumDocs', 'DocIds', 'Topic1-gram',
                                                            'Topic2-gram', 'Topic3-gram', 'TopicN-gram'])
                path = os.path.join(self.output_path,
                                    self.args.case_name + '_' + approach + '_topic_words.csv')
                cluster_df.to_csv(path, encoding='utf-8', index=False)
                # # # Write to a json file
                path = os.path.join(self.output_path,
                                    self.args.case_name + '_' + approach + '_topic_words.json')
                cluster_df.to_json(path, orient='records')
                print('Output topics per cluster to ' + path)
        except Exception as err:
            print("Error occurred! {err}".format(err=err))


# Main entry
if __name__ == '__main__':
    docCluster = DocumentCluster()
    # docCluster.get_sentence_embedding()
    # docCluster.cluster_doc_by_hdbscan()
    # docCluster.cluster_doc_by_KMeans()
    # TopicUtility.visualise_cluster_results(docCluster.args.min_cluster_size)
    # docCluster.collect_tf_idf_terms_by_cluster()
    docCluster.derive_topic_words_from_cluster_docs()

# # Derive topic words using TF-IDF
# topic_words_tf_idf = TopicUtility.derive_topic_words_tf_idf(tf_idf_df, doc_ids)
# We use the number of TF-IDF terms as the limitation
# max_length = len(topic_words_tf_idf)
# Collect the result
# Derive key words spanning across the articles in a cluster
# topic_bag_of_words = TopicUtility.derive_bag_of_words(doc_ids, doc_texts)
# Derive topic words through collocation likelihood
# topic_words_collocations = TopicUtility.derive_topic_words_using_collocations('likelihood',
#                                                                               doc_ids,
#                                                                               doc_texts)
# def cluster_doc_by_agglomerative(self):
#     # Normalize the embeddings to unit length
#     # corpus_embeddings = self.document_embeddings / np.linalg.norm(self.document_embeddings, axis=1, keepdims=True)
#     # Perform Agglomerative clustering
#     clustering_model = AgglomerativeClustering(n_clusters=None, distance_threshold=1.5)
#     clustering_model.fit(self.clusterable_embedding)
#     clusters = clustering_model.labels_
#     self.result_df['Agglomerative_Cluster'] = clusters
