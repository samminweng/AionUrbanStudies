import os
import re
import string
from argparse import Namespace
# Obtain the cluster results of the best results and extract cluster topics using TF-IDF
from pathlib import Path

import gensim
from gensim import corpora
from gensim.models import Phrases, CoherenceModel
from nltk import WordNetLemmatizer
from nltk.tokenize import word_tokenize, sent_tokenize
import pandas as pd
from BERTModelDocClusterUtility import BERTModelDocClusterUtility

# Ref: https://radimrehurek.com/gensim/auto_examples/tutorials/run_lda.html#sphx-glr-auto-examples-tutorials-run-lda-py
from ClusterTopicUtility import ClusterTopicUtility


class ClusterTopicLDA:
    def __init__(self, _last_iteration):
        self.args = Namespace(
            case_name='CultureUrbanStudyCorpus',
            approach='LDA',
            last_iteration=_last_iteration,
            NUM_TOPICS=5,
            passes=100,
            iterations=400,
            chunksize=10,
            eval_every=None  # Don't evaluate model perplexity, takes too much time.
        )

    # Derive the topic from each cluster of documents using LDA Topic modeling
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
            n_grams = list()
            for text in texts:
                candidates = list()
                cleaned_text = BERTModelDocClusterUtility.preprocess_text(text)
                sentences = sent_tokenize(cleaned_text)
                uni_grams = ClusterTopicUtility.generate_n_gram_candidates(sentences, 1)
                bi_grams = ClusterTopicUtility.generate_n_gram_candidates(sentences, 2)
                tri_grams = ClusterTopicUtility.generate_n_gram_candidates(sentences, 3)
                candidates.extend(uni_grams)
                candidates.extend(bi_grams)
                candidates.extend(tri_grams)
                n_grams.append(candidates)

            df['Ngrams'] = n_grams
            # Group the documents and doc_id by clusters
            docs_per_cluster_df = df.groupby(['Cluster'], as_index=False) \
                .agg({'DocId': lambda doc_id: list(doc_id), 'Ngrams': lambda n_grams: list(n_grams)})
            total = len(df)
            for num_topics in range(3, 11):
                results = list()
                for i, cluster in docs_per_cluster_df.iterrows():
                    try:
                        cluster_no = cluster['Cluster']
                        n_grams = cluster['Ngrams']
                        # Create a dictionary
                        dictionary = corpora.Dictionary(n_grams)
                        corpus = [dictionary.doc2bow(n_gram) for n_gram in n_grams]
                        # Build the LDA model
                        ldamodel = gensim.models.ldamodel.LdaModel(corpus, num_topics=num_topics,
                                                                   id2word=dictionary, passes=self.args.passes,
                                                                   iterations=self.args.iterations,
                                                                   eval_every=self.args.eval_every,
                                                                   chunksize=self.args.chunksize)
                        top_topics = ldamodel.top_topics(corpus, topn=10)

                        cm = CoherenceModel(model=ldamodel, corpus=corpus, dictionary=dictionary,
                                            coherence='u_mass')
                        avg_topic_coherence = sum([t[1] for t in top_topics]) / num_topics
                        print('Average topic coherence: %.4f.' % avg_topic_coherence)
                        topic_score = cm.get_coherence()
                        # Collect all the topic words
                        lda_topics = list()
                        for topic in top_topics:
                            topic_words = list(map(lambda t: t[1], topic[0]))
                            lda_topics.append({
                                'topic': topic_words,
                                'score': topic[1]  # Topic Coherence score
                            })
                        num_docs = len(cluster['DocId'])
                        percent = num_docs / total
                        results.append({
                            "NumTopics": num_topics,
                            "Cluster": cluster_no,
                            'DocId': cluster['DocId'],
                            'NumDocs': len(cluster['DocId']),
                            'Percent': round(percent, 3),
                            "Score": topic_score,
                            "LDATopics": lda_topics,
                        })
                    except Exception as _err:
                        print("Error occurred! {err}".format(err=_err))
                # Write the result to csv and json file
                cluster_df = pd.DataFrame(results,
                                          columns=['NumTopics', 'Cluster', 'NumDocs', 'Percent', 'DocId', "Score",
                                                   'LDATopics'])
                topic_folder = os.path.join('output', self.args.case_name, 'LDA_topics', 'experiments')
                Path(topic_folder).mkdir(parents=True, exist_ok=True)
                # # # Write to a json file
                path = os.path.join(topic_folder,
                                    self.args.case_name + '_LDA_cluster_topics_#' + str(num_topics) + '.json')
                cluster_df.to_json(path, orient='records')
                # Write to a csv file
                path = os.path.join(topic_folder,
                                    self.args.case_name + '_LDA_cluster_topics#' + str(num_topics) + '.csv')
                cluster_df.to_csv(path, encoding='utf-8', index=False)
                # Write a summary
                for i in range(0, num_topics):
                    cluster_df['LDATopics#' + str(i)] = cluster_df.apply(lambda c: c['LDATopics'][i]['topic'], axis=1)
                cluster_df.drop('LDATopics', axis=1, inplace=True)
                path = os.path.join(topic_folder,
                                    self.args.case_name + '_LDA_cluster_topic_summary#' + str(num_topics) + '.csv')
                cluster_df.to_csv(path, encoding='utf-8', index=False)
                print('Output topics per cluster to ' + path)
        except Exception as err:
            print("Error occurred! {err}".format(err=err))

    def get_best_cluster_topics_by_LDA(self):
        try:
            folder = os.path.join('output', self.args.case_name, 'LDA_topics', 'experiments')
            results = list()
            for num_topics in range(3, 11):
                path = os.path.join(folder, self.args.case_name + '_LDA_cluster_topics_#' + str(num_topics) + '.json')
                # Load the documents clustered by
                df = pd.read_json(path)
                cluster_topics = df.to_dict("records")
                if len(results) == 0:
                    results = cluster_topics
                else:
                    for i in range(len(results)):
                        result = results[i]
                        cluster_no = result['Cluster']
                        cluster_topic = next(ct for ct in cluster_topics if ct['Cluster'] == cluster_no)
                        if cluster_topic['Score'] > result['Score']:
                            results[i] = cluster_topic
            topic_folder = os.path.join('output', self.args.case_name, 'LDA_topics', )
            Path(topic_folder).mkdir(parents=True, exist_ok=True)
            cluster_df = pd.DataFrame(results,
                                      columns=['Cluster', 'NumDocs', 'Percent', 'DocId', 'NumTopics', "Score",
                                               'LDATopics'])
            # # # Write to a json file
            path = os.path.join(topic_folder,
                                self.args.case_name + '_LDA_cluster_topics.json')
            cluster_df.to_json(path, orient='records')
            # Write to a csv file
            path = os.path.join(topic_folder,
                                self.args.case_name + '_LDA_cluster_topics.csv')
            cluster_df.to_csv(path, encoding='utf-8', index=False)
        except Exception as err:
            print("Error occurred! {err}".format(err=err))


# Main entry
if __name__ == '__main__':
    last_iteration = 10
    ct = ClusterTopicLDA(10)
    ct.derive_cluster_topics_by_LDA()
    ct.get_best_cluster_topics_by_LDA()
