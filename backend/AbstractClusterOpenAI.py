# Cluster the document using OpenAI model
# Ref: https://openai.com/blog/introducing-text-and-code-embeddings/
import os
import sys
from argparse import Namespace
from functools import reduce
from pathlib import Path

import hdbscan
import umap
from nltk.tokenize import sent_tokenize, word_tokenize
import pandas as pd
import openai, numpy as np

from sklearn.metrics import pairwise_distances, silhouette_samples

openai.organization = "org-yZnUvR0z247w0HQoS6bMJ0WI"
openai.api_key = os.getenv("OPENAI_API_KEY")


# print(openai.Model.list())


class AbstractClusterOpenAI:
    def __init__(self, _iteration):
        self.args = Namespace(
            case_name='AIMLUrbanStudyCorpus',
            embedding_name='OpenAIEmbedding',
            model_name='curie',
            iteration=_iteration,
            iteration_folder='iteration_' + str(_iteration),
            phase='abstract_clustering_phase',
            path='data',
            n_neighbors=150,
            min_dist=0.0,
            epilson=0.0,
            dimensions=[500, 450, 400, 350, 300, 250, 200, 150, 100, 95, 90, 85, 80, 75, 70, 65, 60, 55,
                        50, 45, 40, 35, 30, 25, 20],
            min_samples=15,
            # min_samples=[30, 25, 20, 15, 10],
            min_cluster_size=[10, 15, 20, 25, 30, 35, 40, 45, 50]
        )
        path = os.path.join('data', self.args.case_name, self.args.case_name + '_cleaned.csv')
        self.text_df = pd.read_csv(path)
        # # # # # Load all document vectors without outliers
        self.text_df['Text'] = self.text_df['Title'] + ". " + self.text_df['Abstract']
        # Filter out dimensions > the length of text df
        self.args.dimensions = list(filter(lambda d: d < len(self.text_df) - 5 and d != 768, self.args.dimensions))

    # Get doc vectors from OpenAI embedding API
    def get_doc_vectors(self, is_load=False):
        def clean_sentence(_sentences):
            # Preprocess the sentence
            cleaned_sentences = list()  # Skip copy right sentence
            for sentence in _sentences:
                if u"\u00A9" not in sentence.lower() and 'licensee' not in sentence.lower() \
                        and 'copyright' not in sentence.lower() and 'rights reserved' not in sentence.lower():
                    try:
                        cleaned_words = word_tokenize(sentence.lower())
                        # Keep alphabetic characters only and remove the punctuation
                        cleaned_sentences.append(" ".join(cleaned_words))  # merge tokenized words into sentence
                    except Exception as _err:
                        print("Error occurred! {err}".format(err=_err))
            return cleaned_sentences

        try:
            # Collect all the texts
            cleaned_texts = list()
            # Search all the subject words
            for i, row in self.text_df.iterrows():
                try:
                    sentences = clean_sentence(sent_tokenize(row['Text']))  # Clean the sentences
                    cleaned_text = " ".join(sentences)
                    cleaned_texts.append(cleaned_text)
                except Exception as _err:
                    print("Error occurred! {err}".format(err=_err))
            self.text_df['CleanText'] = cleaned_texts
            resp = openai.Embedding.create(
                input=cleaned_texts,
                engine="text-similarity-" + self.args.model_name + "-001")
            doc_embeddings = list()
            for doc_embedding in resp['data']:
                doc_embeddings.append(doc_embedding['embedding'])
            print(doc_embeddings)
            self.text_df['DocVectors'] = doc_embeddings
            # Print out the doc vector
            print(self.text_df)
            folder = os.path.join('output', self.args.case_name + '_' + self.args.embedding_name,
                                  'abstract_clustering_phase', self.args.iteration_folder, 'doc_vectors')
            Path(folder).mkdir(parents=True, exist_ok=True)
            path = os.path.join(folder, 'doc_vectors.json')
            self.text_df.to_json(path, orient='records')
        except Exception as err:
            print("Error occurred! {err}".format(err=err))

    # Experiment UMAP + HDBSCAN clustering and evaluate the clustering results with 'Silhouette score'
    def run_HDBSCAN_cluster_experiments(self):
        # Calculate Silhouette score
        # Ref: https://towardsdatascience.com/silhouette-coefficient-validating-clustering-techniques-e976bb81d10c
        # Ref: https://scikit-learn.org/stable/modules/generated/sklearn.metrics.silhouette_score.html
        def compute_Silhouette_score(_cluster_labels, _cluster_vectors, _cluster_results):
            # score = 1 indicates good clusters that each cluster distinguishes from other clusters
            # score = 0 no difference between clusters
            # score = -1 clusters are wrong
            try:
                # start = datetime.now()
                # Get silhouette score for each cluster
                silhouette_scores = silhouette_samples(_cluster_vectors, _cluster_labels, metric='cosine')
                avg_scores = list()
                # Get each individual cluster's score
                for _cluster_result in _cluster_results:
                    cluster = _cluster_result['cluster']
                    cluster_silhouette_scores = silhouette_scores[np.array(cluster_labels) == cluster]
                    cluster_score = np.mean(cluster_silhouette_scores)
                    _cluster_result['score'] = cluster_score
                    avg_scores.append(cluster_score)
                    # end = datetime.now()
                avg_scores = np.mean(avg_scores)
                # difference = (end - start).total_seconds()
                # print("Time difference {d} second".format(d=difference))
                return _cluster_results, avg_scores
            except Exception as err:
                print("Error occurred! {err}".format(err=err))
                return -1

        # Collect clustering results and find outliers and the cluster of minimal size
        def collect_cluster_results(_doc_vectors, _cluster_labels):
            try:
                _results = list()
                for _doc, _label in zip(_doc_vectors, _cluster_labels):
                    _doc_id = _doc['DocId']
                    _found = next((r for r in _results if r['cluster'] == _label), None)
                    if not _found:
                        _results.append({'cluster': _label, 'doc_ids': [_doc_id]})
                    else:
                        _found['doc_ids'].append(_doc_id)
                _results = sorted(_results, key=lambda c: c['cluster'], reverse=True)
                # Add the count
                for _result in _results:
                    _result['count'] = len(_result['doc_ids'])
                    _result['doc_ids'] = _result['doc_ids']
                return _results
            except Exception as c_err:
                print("Error occurred! {err}".format(err=c_err))

        # Load doc vectors
        folder = os.path.join('output', self.args.case_name + '_' + self.args.embedding_name,
                              self.args.phase, self.args.iteration_folder, 'doc_vectors')
        path = os.path.join(folder, 'doc_vectors.json')
        doc_vector_df = pd.read_json(path)
        doc_vectors = doc_vector_df.to_dict("records")
        # Doc vectors from OpenAI is 4,096
        print("OpenAI dimension {d}".format(d=len(doc_vector_df['DocVectors'].tolist()[0])))
        # Experiment HDBSCAN clustering with different parameters
        results = list()
        max_score = 0.0
        best_reduced_vectors = None
        for dimension in self.args.dimensions:
            if dimension <= 500:
                # Run HDBSCAN on reduced dimensional vectors
                reduced_vectors = umap.UMAP(
                    n_neighbors=self.args.n_neighbors,
                    min_dist=self.args.min_dist,
                    n_components=dimension,
                    random_state=42,
                    metric="cosine").fit_transform(doc_vector_df['DocVectors'].tolist())
            else:
                # Run HDBSCAN on raw vectors
                reduced_vectors = np.vstack(doc_vector_df['DocVectors'])  # Convert to 2D numpy array
            # print(reduced_vectors)
            # for min_samples in self.args.min_samples:
            epsilon = self.args.epilson
            min_samples = self.args.min_samples
            for min_cluster_size in self.args.min_cluster_size:
                result = {'dimension': dimension,
                          'min_cluster_size': min_cluster_size,
                          'avg_score': None, 'total_clusters': None,
                          }
                try:
                    # Compute the cosine distance/similarity for each doc vectors
                    distances = pairwise_distances(reduced_vectors, metric='cosine')
                    # Cluster reduced vectors using HDBSCAN
                    cluster_labels = hdbscan.HDBSCAN(min_cluster_size=min_cluster_size,
                                                     min_samples=min_samples,
                                                     cluster_selection_epsilon=epsilon,
                                                     metric='precomputed').fit_predict(
                        distances.astype('float64')).tolist()
                    # Aggregate the cluster results
                    cluster_results = collect_cluster_results(doc_vectors, cluster_labels)
                    # Sort cluster result by count

                    # Compute silhouette score for clustered results
                    distance_vectors = distances.tolist()
                    if len(cluster_results) > 0:
                        cluster_results, avg_score = compute_Silhouette_score(cluster_labels, distance_vectors, cluster_results)
                        outlier = next(r for r in cluster_results if r['cluster'] == -1)
                        result['avg_score'] = avg_score
                        result['total_clusters'] = len(cluster_results)
                        result['outlier'] = outlier['count']
                        result['cluster_results'] = cluster_results
                        if max_score <= avg_score:
                            result['reduced_vectors'] = reduced_vectors.tolist()
                            max_score = avg_score
                        results.append(result)
                except Exception as _err:
                    print("Error occurred! {err}".format(err=_err))
                    sys.exit(-1)
            # print(result)

        # Output the clustering results of a dimension
        folder = os.path.join('output', self.args.case_name + '_' + self.args.embedding_name, self.args.phase,
                              self.args.iteration_folder, 'hdbscan_experiments')
        Path(folder).mkdir(parents=True, exist_ok=True)
        # Output the detailed clustering results
        result_df = pd.DataFrame(results)
        # Output cluster results to CSV
        path = os.path.join(folder, 'HDBSCAN_cluster_doc_vector_results.csv')
        result_df.to_csv(path, encoding='utf-8', index=False, columns=['dimension', 'min_cluster_size',
                                                                       'avg_score', 'total_clusters', 'outlier', 'cluster_results'])
        path = os.path.join(folder, 'HDBSCAN_cluster_doc_vector_results.json')
        result_df.to_json(path, orient='records')




# Main entry
if __name__ == '__main__':
    try:
        # Re-cluster large cluster into sub-clusters
        iteration = 1
        ac = AbstractClusterOpenAI(iteration)
        # ac.get_doc_vectors()
        ac.run_HDBSCAN_cluster_experiments()

    except Exception as err:
        print("Error occurred! {err}".format(err=err))
