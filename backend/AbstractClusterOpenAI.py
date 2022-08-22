# Cluster the document using OpenAI model
# Ref: https://openai.com/blog/introducing-text-and-code-embeddings/
import os
from argparse import Namespace
from pathlib import Path

from nltk.tokenize import sent_tokenize, word_tokenize
import pandas as pd
import openai, numpy as np
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
            path='data',
            n_neighbors=150,
            min_dist=0.0,
            dimensions=[768, 200, 150, 100, 95, 90, 85, 80, 75, 70, 65, 60, 55, 50, 45, 40, 35, 30, 25, 20],
            min_samples=[30, 25, 20, 15, 10],
            min_cluster_size=[100, 80, 50]
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
        # Load doc vectors
        folder = os.path.join('output', self.args.case_name + '_' + self.args.embedding_name,
                              'abstract_clustering_phase', self.args.iteration_folder, 'doc_vectors')
        path = os.path.join(folder, 'doc_vectors.json')
        doc_vectors = pd.read_json(path).to_dict("records")
        print(doc_vectors)



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
