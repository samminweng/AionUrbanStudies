import os
import re
import string
from argparse import Namespace
# Obtain the cluster results of the best results and extract cluster topics using TF-IDF
from pathlib import Path

import gensim
from gensim import corpora
from gensim.models import Phrases
from nltk import WordNetLemmatizer
from nltk.tokenize import word_tokenize

import pandas as pd

from BERTModelDocClusterUtility import BERTModelDocClusterUtility


# Ref: https://towardsdatascience.com/topic-modelling-in-python-with-nltk-and-gensim-4ef03213cd21
class ClusterTopicLDA:
    def __init__(self, _last_iteration):
        self.args = Namespace(
            case_name='CultureUrbanStudyCorpus',
            approach='LDA',
            last_iteration=_last_iteration,
            NUM_TOPICS=10,
            passes=20
        )

    # Derive the topic from each cluster of documents using
    def derive_cluster_topics_by_LDA(self):
        # approach = 'HDBSCAN_Cluster'
        try:

            path = os.path.join('output', self.args.case_name, self.args.case_name + '_clusters.json')
            # Load the documents clustered by
            df = pd.read_json(path)
            # Update text column
            df['Text'] = df['Title'] + ". " + df['Abstract']
            texts = df['Text'].tolist()
            # Preprocess the texts
            texts = list(map(lambda text: BERTModelDocClusterUtility.preprocess_text(text), texts))
            # Remove punctuation
            texts = list(map(lambda text: text.translate(str.maketrans('', '', string.punctuation + "’")), texts))
            # Tokenize the text
            texts = list(map(lambda text: word_tokenize(text), texts))
            # Remove stop words
            texts = list(map(lambda tokens: [token for token in tokens if token.lower() not in BERTModelDocClusterUtility.stop_words], texts))
            # Lemmatize
            lemmatizer = WordNetLemmatizer()
            texts = [[lemmatizer.lemmatize(token) for token in text] for text in texts]
            # Add bigram
            bigram = Phrases(texts, min_count=20)
            for idx in range(len(texts)):
                for token in bigram[texts[idx]]:
                    if '_' in token:
                        # Token is a bigram, add to document.
                        texts[idx].append(token)
            df['Text'] = texts
            # Group the documents and doc_id by clusters
            docs_per_cluster_df = df.groupby(['Cluster'], as_index=False) \
                .agg({'DocId': lambda doc_id: list(doc_id), 'Text': lambda text: list(text)})
            total = len(df)
            results = list()
            for i, cluster in docs_per_cluster_df.iterrows():
                try:
                    cluster_no = cluster['Cluster']
                    doc_texts = cluster['Text']
                    # Create a dictionary
                    dictionary = corpora.Dictionary(doc_texts)
                    corpus = [dictionary.doc2bow(text) for text in doc_texts]
                    # Build the LDA model
                    ldamodel = gensim.models.ldamodel.LdaModel(corpus, num_topics=self.args.NUM_TOPICS,
                                                               id2word=dictionary, passes=self.args.passes)
                    top_topics = ldamodel.top_topics(corpus, topn=5)
                    avg_topic_coherence = sum([t[1] for t in top_topics]) / self.args.NUM_TOPICS
                    print('Average topic coherence: %.4f.' % avg_topic_coherence)
                    lda_topics = list()
                    for topic in top_topics:
                        topic_words = list(map(lambda t: t[1], topic[0]))
                        lda_topics.append({
                            'topic': topic_words,
                            'score': topic[1]   # Topic Coherence score
                        })
                    num_docs = len(cluster['DocId'])
                    percent = 100 * num_docs/total
                    results.append({
                        "Cluster": cluster_no,
                        'DocId': cluster['DocId'],
                        'Num Docs': len(cluster['DocId']),
                        'Percent': round(percent, 2),
                        "LDATopics": lda_topics
                    })
                except Exception as _err:
                    print("Error occurred! {err}".format(err=_err))
            # Write the result to csv and json file
            cluster_df = pd.DataFrame(results, columns=['Cluster', 'Num Docs', 'Percent', 'DocId', 'LDATopics'])
            topic_folder = os.path.join('output', self.args.case_name, 'LDA_topics')
            Path(topic_folder).mkdir(parents=True, exist_ok=True)
            # # # Write to a json file
            path = os.path.join(topic_folder, self.args.case_name + '_LDA_cluster_topics.json')
            cluster_df.to_json(path, orient='records')
            # Write to a csv file
            path = os.path.join(topic_folder, self.args.case_name + '_LDA_cluster_topics.csv')
            cluster_df.to_csv(path, encoding='utf-8', index=False)
            # Write a summary
            cluster_df['LDATopics'] = cluster_df.apply(lambda c:
                                                       list(map(lambda t: "(" + ", ".join(t['topic']) + ")", c['LDATopics'])), axis=1)
            path = os.path.join(topic_folder, self.args.case_name + '_LDA_cluster_topic_summary.csv')
            cluster_df.to_csv(path, encoding='utf-8', index=False)
            print('Output topics per cluster to ' + path)
        except Exception as err:
            print("Error occurred! {err}".format(err=err))


# Main entry
if __name__ == '__main__':
    last_iteration = 10
    ct = ClusterTopicLDA(10)
    ct.derive_cluster_topics_by_LDA()
