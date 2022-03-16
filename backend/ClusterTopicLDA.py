import os
from argparse import Namespace
# Obtain the cluster results of the best results and extract cluster topics using TF-IDF
from pathlib import Path
import gensim
from gensim import corpora
from gensim.models import Phrases
from nltk.tokenize import sent_tokenize
import pandas as pd
from BERTModelDocClusterUtility import BERTModelDocClusterUtility

# Ref: https://radimrehurek.com/gensim/auto_examples/tutorials/run_lda.html#sphx-glr-auto-examples-tutorials-run-lda-py
from ClusterTopicUtility import ClusterTopicUtility


class ClusterTopicLDA:
    def __init__(self):
        # self.cluster_no = _cluster_no
        self.args = Namespace(
            # case_name='CultureUrbanStudyCorpus',
            case_name='AIMLUrbanStudyCorpus',
            approach='LDA',
            passes=100,
            iterations=400,
            chunksize=10,
            eval_every=None,  # Don't evaluate model perplexity, takes too much time.
            folder='iteration',
            # folder='cluster_' + str(_cluster_no),
        )
        # Load Key phrase
        path = os.path.join('output', self.args.case_name, self.args.folder, 'key_phrases',
                            self.args.case_name + '_cluster_terms_key_phrases.json')
        self.cluster_key_phrases_df = pd.read_json(path)
        # Sort by Cluster
        self.cluster_key_phrase_df = self.cluster_key_phrases_df.sort_values(by=['Cluster'], ascending=True)

    # Derive n_gram from each individual paper
    def derive_n_grams_group_by_clusters(self):
        try:
            path = os.path.join('output', self.args.case_name, self.args.folder,
                                self.args.case_name + '_clusters.json')
            # Load the documents clustered by
            df = pd.read_json(path)
            # Update text column
            df['Text'] = df['Title'] + ". " + df['Abstract']
            texts = df['Text'].tolist()
            # Preprocess the texts
            n_gram_list = list()
            for text in texts:
                candidates = list()
                cleaned_text = BERTModelDocClusterUtility.preprocess_text(text)
                sentences = sent_tokenize(cleaned_text)
                uni_grams = ClusterTopicUtility.generate_n_gram_candidates(sentences, 1, is_check=False)
                bi_grams = ClusterTopicUtility.generate_n_gram_candidates(sentences, 2, is_check=False)
                tri_grams = ClusterTopicUtility.generate_n_gram_candidates(sentences, 3, is_check=False)
                candidates.extend(uni_grams)
                candidates.extend(bi_grams)
                candidates.extend(tri_grams)
                n_gram_list.append(candidates)
            df['Ngrams'] = n_gram_list
            # Group the n-grams by clusters
            docs_per_cluster_df = df.groupby(['Cluster'], as_index=False) \
                .agg({'DocId': lambda doc_id: list(doc_id), 'Ngrams': lambda n_grams: list(n_grams)})

            # Sort by Cluster
            docs_per_cluster_df = docs_per_cluster_df.sort_values(by=['Cluster'], ascending=True)
            # Load the key phrases
            docs_per_cluster_df['KeyPhrases'] = self.cluster_key_phrases_df['KeyPhrases'].tolist()
            # Reorder the column
            docs_per_cluster_df = docs_per_cluster_df[['Cluster', 'KeyPhrases', 'DocId', 'Ngrams']]
            # Write n_gram to csv and json file
            folder = os.path.join('output', self.args.case_name, self.args.folder,
                                  'LDA_topics', 'n_grams')
            Path(folder).mkdir(parents=True, exist_ok=True)
            path = os.path.join(folder, self.args.case_name + '_doc_n_grams.csv')
            docs_per_cluster_df.to_csv(path, index=False, encoding='utf-8')
            path = os.path.join(folder, self.args.case_name + '_doc_n_grams.json')
            docs_per_cluster_df.to_json(path, orient='records')
            print('Output key phrases per doc to ' + path)
        except Exception as err:
            print("Error occurred! {err}".format(err=err))

    # Derive the topic from each cluster of documents using LDA Topic modeling
    def derive_cluster_topics_by_LDA(self):
        try:
            path = os.path.join('output', self.args.case_name, self.args.folder,
                                'LDA_topics', 'n_grams', self.args.case_name + '_doc_n_grams.json')
            # Load the documents clustered by
            df = pd.read_json(path)
            # Collect
            results = list()
            n_topic = 5
            # Apply LDA Topic model on each cluster of papers
            for i, cluster in df.iterrows():
                try:
                    num_topics = len(cluster['KeyPhrases'])  # Get the number of grouped phrases
                    doc_n_gram_list = cluster['Ngrams']
                    doc_id_list = cluster['DocId']
                    doc_n_grams = tuple(zip(doc_id_list, doc_n_gram_list))
                    # Create a dictionary
                    dictionary = corpora.Dictionary(doc_n_gram_list)
                    corpus = [dictionary.doc2bow(n_gram) for n_gram in doc_n_gram_list]
                    # Build the LDA model
                    ldamodel = gensim.models.ldamodel.LdaModel(corpus, num_topics=num_topics,
                                                               id2word=dictionary, passes=self.args.passes,
                                                               iterations=self.args.iterations,
                                                               eval_every=self.args.eval_every,
                                                               chunksize=self.args.chunksize)
                    top_topic_list = ldamodel.top_topics(corpus, topn=n_topic)
                    total_score = 0
                    # Collect all the topic words
                    lda_topics = list()
                    for topic in top_topic_list:
                        topic_words = list(map(lambda t: t[1], topic[0]))
                        topic_coherence_score, word_docs = ClusterTopicUtility.compute_topic_coherence_score(
                            doc_n_grams, topic_words)
                        lda_topics.append({
                            'topic_words': topic_words,
                            'score': round(topic_coherence_score, 3),  # Topic Coherence score
                            'word_docIds': word_docs
                        })
                        total_score += topic_coherence_score
                    avg_score = total_score / (num_topics * 1.0)
                    # Add one record
                    results.append({
                        "Cluster": cluster['Cluster'],
                        "NumTopics": num_topics,
                        "LDAScore": round(avg_score, 3),
                        "LDATopics": lda_topics,
                        "LDATopic_Words": list(map(lambda topic: (topic['topic_words'], topic['score']), lda_topics))
                    })
                except Exception as _err:
                    print("Error occurred! {err}".format(err=_err))
            # Write the result to csv and json file
            cluster_df = pd.DataFrame(results,
                                      columns=['Cluster', 'NumTopics', 'LDAScore', 'LDATopics', 'LDATopic_Words'])
            topic_folder = os.path.join('output', self.args.case_name, self.args.folder, 'LDA_topics', 'lda_scores')
            Path(topic_folder).mkdir(parents=True, exist_ok=True)
            # # # Write to a json file
            path = os.path.join(topic_folder,
                                self.args.case_name + '_LDA_topics.json')
            cluster_df.to_json(path, orient='records')
            # Write to a csv file
            path = os.path.join(topic_folder,
                                self.args.case_name + '_LDA_topics.csv')
            cluster_df.to_csv(path, encoding='utf-8', index=False)
            print('Output topics per cluster to ' + path)
        except Exception as err:
            print("Error occurred! {err}".format(err=err))

    # Compute the score
    def compute_key_phrase_scores(self):
        try:
            # Load documents
            path = os.path.join('output', self.args.case_name, self.args.folder,
                                self.args.case_name + '_clusters.json')
            docs = pd.read_json(path).to_dict("records")
            print(docs)
            # Load n-grams
            path = os.path.join('output', self.args.case_name, self.args.folder, 'LDA_topics', 'n_grams',
                                self.args.case_name + '_doc_n_grams.json')
            # Load the documents clustered by
            clusters = pd.read_json(path).to_dict("records")
            # Store the phrase scores
            results = list()
            # Get the cluster
            for cluster in clusters:
                doc_n_gram_list = cluster['Ngrams']
                doc_id_list = cluster['DocId']
                doc_n_grams = tuple(zip(doc_id_list, doc_n_gram_list))
                total_score = 0
                key_phrase_groups = list()
                for kp_group in cluster['KeyPhrases']:
                    topic_words = ClusterTopicUtility.collect_topic_words_from_key_phrases(kp_group['Key-phrases'],
                                                                                           doc_n_grams)
                    # print(topic_words)
                    # Topic coherence score
                    score, word_docs = ClusterTopicUtility.compute_topic_coherence_score(doc_n_grams, topic_words)
                    key_phrase_group = {"topic_words": topic_words, 'score': round(score, 3), 'word_docIds': word_docs,
                                        'key-phrases': kp_group['Key-phrases'], 'NumDocs': kp_group['NumDocs'],
                                        'DocIds': kp_group['DocIds']}
                    total_score += score
                    key_phrase_groups.append(key_phrase_group)
                num_topics = len(cluster['KeyPhrases'])
                avg_score = total_score / (num_topics * 1.0)
                # Add one record
                results.append({
                    "Cluster": cluster['Cluster'],
                    "NumTopics": num_topics,
                    "KeyPhraseScore": round(avg_score, 3),
                    "KeyPhrases": key_phrase_groups,
                    "KeyPhrase_Words": list(map(lambda topic: (topic['topic_words'], topic['score']), key_phrase_groups))
                })
            # Write the updated grouped key phrases
            cluster_df = pd.DataFrame(results,
                                      columns=['Cluster', 'NumTopics', 'KeyPhraseScore', 'KeyPhrase_Words', 'KeyPhrases'])
            folder = os.path.join('output', self.args.case_name, self.args.folder, 'LDA_topics', 'key_phrase_scores')
            Path(folder).mkdir(parents=True, exist_ok=True)
            # # # Write to a json file
            path = os.path.join(folder, self.args.case_name + '_key_phrase_scores.json')
            cluster_df.to_json(path, orient='records')
            # Write to a csv file
            path = os.path.join(folder, self.args.case_name + '_key_phrase_scores.csv')
            cluster_df.to_csv(path, encoding='utf-8', index=False)
            print('Output phrase scores to ' + path)
        except Exception as err:
            print("Error occurred! {err}".format(err=err))

    # Combine LDA Cluster topics with grouped key phrase results
    def combine_LDA_topics_key_phrase_to_file(self):
        try:
            # # Load key phrase scores
            folder = os.path.join('output', self.args.case_name, self.args.folder, 'key_phrases')
            path = os.path.join(folder, self.args.case_name + '_cluster_terms_key_phrases.json')
            cluster_df = pd.read_json(path)
            # Load results of LDA Topic model
            folder = os.path.join('output', self.args.case_name, self.args.folder, 'LDA_topics', 'lda_scores')
            path = os.path.join(folder, self.args.case_name + '_LDA_topics.json')
            lda_topics_df = pd.read_json(path)
            # # # Load cluster topic, key phrases
            cluster_df['NumTopics'] = lda_topics_df['NumTopics'].tolist()
            cluster_df['LDATopics'] = lda_topics_df['LDATopics'].tolist()
            cluster_df['LDAScore'] = lda_topics_df['LDAScore'].tolist()
            # Load results of key phrase groups
            folder = os.path.join('output', self.args.case_name, self.args.folder, 'LDA_topics', 'key_phrase_scores')
            path = os.path.join(folder, self.args.case_name + '_key_phrase_scores.json')
            key_phrase_groups_df = pd.read_json(path)
            cluster_df['KeyPhrases'] = key_phrase_groups_df['KeyPhrases']
            cluster_df['KeyPhraseScore'] = key_phrase_groups_df['KeyPhraseScore']
            # Compute the percent
            total = cluster_df['NumDocs'].sum()
            cluster_df['Percent'] = cluster_df['NumDocs'].apply(lambda x: x / total)
            # Output the overall results
            df = cluster_df[['Cluster', 'NumDocs', 'Percent', 'DocIds', 'Terms',
                             'KeyPhraseScore', 'KeyPhrases', 'LDAScore', 'LDATopics']]
            # # # # Write to a json file
            folder = os.path.join('output', self.args.case_name, self.args.folder)
            path = os.path.join(folder, self.args.case_name + '_cluster_terms_key_phrases_LDA_topics.json')
            df.to_json(path, orient='records')
            # Write to a csv file
            path = os.path.join(folder, self.args.case_name + '_cluster_terms_key_phrases_LDA_topics.csv')
            df.to_csv(path, encoding='utf-8', index=False)
        except Exception as err:
            print("Error occurred! {err}".format(err=err))


# Main entry
if __name__ == '__main__':
    try:
        # _cluster_no = 2
        # ct = ClusterTopicLDA(_cluster_no)
        ct = ClusterTopicLDA()
        ct.derive_n_grams_group_by_clusters()
        ct.derive_cluster_topics_by_LDA()
        ct.compute_key_phrase_scores()
        ct.combine_LDA_topics_key_phrase_to_file()
    except Exception as err:
        print("Error occurred! {err}".format(err=err))
