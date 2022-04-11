import os
from argparse import Namespace
from pathlib import Path
import gensim
from gensim import corpora
from gensim.models import Phrases
from nltk.tokenize import sent_tokenize
import pandas as pd

# Find topic words from each keyword cluster
from BERTArticleClusterUtility import BERTArticleClusterUtility
from TopicKeywordClusterUtility import TopicKeywordClusterUtility


# Obtain the important words from each keyword cluster
class TopicKeywordCluster:
    def __init__(self):
        self.args = Namespace(
            case_name='AIMLUrbanStudyCorpus',
            passes=100,
            iterations=400,
            chunksize=10,
            eval_every=None,  # Don't evaluate model perplexity, takes too much time.
            folder='cluster_merge'
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
                cleaned_text = BERTArticleClusterUtility.preprocess_text(text)
                sentences = sent_tokenize(cleaned_text)
                uni_grams = TopicKeywordClusterUtility.generate_n_gram_candidates(sentences, 1)
                bi_grams = TopicKeywordClusterUtility.generate_n_gram_candidates(sentences, 2)
                tri_grams = TopicKeywordClusterUtility.generate_n_gram_candidates(sentences, 3)
                candidates = candidates + uni_grams
                candidates = candidates + bi_grams
                candidates = candidates + tri_grams
                n_gram_list.append(candidates)
            df['Ngrams'] = n_gram_list
            # Group the n-grams by clusters
            docs_per_cluster_df = df.groupby(['Cluster'], as_index=False) \
                .agg({'DocId': lambda doc_id: list(doc_id), 'Ngrams': lambda n_grams: list(n_grams)})
            # Sort by Cluster
            docs_per_cluster_df = docs_per_cluster_df.sort_values(by=['Cluster'], ascending=True)
            # Reorder the column
            docs_per_cluster_df = docs_per_cluster_df[['Cluster', 'DocId', 'Ngrams']]
            # Write n_gram to csv and json file
            folder = os.path.join('output', self.args.case_name, self.args.folder,
                                  'topics', 'n_grams')
            Path(folder).mkdir(parents=True, exist_ok=True)
            path = os.path.join(folder, self.args.case_name + '_doc_n_grams.csv')
            docs_per_cluster_df.to_csv(path, index=False, encoding='utf-8')
            path = os.path.join(folder, self.args.case_name + '_doc_n_grams.json')
            docs_per_cluster_df.to_json(path, orient='records')
            print('Output key phrases per doc to ' + path)
        except Exception as err:
            print("Error occurred! {err}".format(err=err))

    # Derive the topic from each cluster of documents using LDA Topic modeling
    # Ref: https://radimrehurek.com/gensim/auto_examples/tutorials/run_lda.html#sphx-glr-auto-examples-tutorials-run-lda-py
    def derive_topics_from_article_cluster_by_LDA(self):
        try:
            path = os.path.join('output', self.args.case_name, self.args.folder,
                                'topics', 'n_grams', self.args.case_name + '_doc_n_grams.json')
            # Load the documents clustered by
            df = pd.read_json(path)
            # Collect
            results = list()
            n_topic = 5
            # Apply LDA Topic model on each cluster of papers
            for i, cluster in df.iterrows():
                try:
                    cluster_no = cluster['Cluster']
                    # Get the keyword clusters for the article cluster
                    article_cluster = self.cluster_key_phrase_df.loc[self.cluster_key_phrase_df['Cluster'] == cluster_no].iloc[0]
                    keyword_clusters = article_cluster['KeyPhrases']
                    num_topics = len(keyword_clusters)  # Get the number of grouped phrases
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
                        topic_coherence_score, word_docs = TopicKeywordClusterUtility.compute_topic_coherence_score(
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
            topic_folder = os.path.join('output', self.args.case_name, self.args.folder, 'topics', 'lda_topics')
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
    def collect_topics_from_keyword_cluster(self):
        try:
            folder = os.path.join('output', self.args.case_name, self.args.folder)
            # Load the documents clustered by
            clusters = self.cluster_key_phrases_df.to_dict("records")
            # Store the phrase scores
            results = list()
            # Get the cluster
            for cluster in clusters:
                cluster_no = cluster['Cluster']
                
                # Load doc_key_phrases for article cluster
                key_phrase_groups = cluster['KeyPhrases']
                for group in key_phrase_groups:
                    topic_words = TopicKeywordClusterUtility.collect_topic_words_from_key_phrases(group['Key-phrases'])
                    # score, word_docs = TopicKeywordClusterUtility.compute_topic_coherence_score(doc_n_grams, topic_words)
                    group["TopicWords"] = topic_words
                #
                num_topics = len(key_phrase_groups)
                # Write output to
                keyword_cluster_folder = os.path.join('output', self.args.case_name, self.args.folder, 'topics',
                                                      'keyword_cluster')
                Path(folder).mkdir(parents=True, exist_ok=True)
                df = pd.DataFrame(key_phrase_groups)
                df = df[['Group', 'TopicWords', 'NumPhrases', 'Key-phrases', 'NumDocs', 'DocIds', 'score',
                         'dimension', 'min_samples', 'min_cluster_size']]
                path = os.path.join(keyword_cluster_folder, 'group_key_phrases_cluster_#' + str(cluster_no) + ".csv")
                df.to_csv(path, encoding='utf-8', index=False)
                key_phrase_groups = df.to_dict("records")
                # Add one record
                results.append({
                    "Cluster": cluster['Cluster'],
                    "NumTopics": num_topics,
                    "KeywordClusters": key_phrase_groups
                })
            # Write the updated grouped key phrases
            cluster_df = pd.DataFrame(results,
                                      columns=['Cluster', 'NumTopics', 'KeywordClusters'])
            folder = os.path.join('output', self.args.case_name, self.args.folder, 'topics')

            # # # Write to a json file
            path = os.path.join(folder, self.args.case_name + '_key_phrase_topics.json')
            cluster_df.to_json(path, orient='records')
            # Write to a csv file
            path = os.path.join(folder, self.args.case_name + '_key_phrase_topics.csv')
            cluster_df.to_csv(path, encoding='utf-8', index=False)
            print('Output phrase scores to ' + path)
        except Exception as err:
            print("Error occurred! {err}".format(err=err))

    # Combine LDA Cluster topics with grouped key phrase results
    def combine_topics_keyword_cluster_to_file(self):
        try:
            # # Load key phrase scores
            folder = os.path.join('output', self.args.case_name, self.args.folder, 'key_phrases')
            path = os.path.join(folder, self.args.case_name + '_cluster_terms_key_phrases.json')
            cluster_df = pd.read_json(path)
            # Load results of key phrase groups
            folder = os.path.join('output', self.args.case_name, self.args.folder, 'topics')
            path = os.path.join(folder, self.args.case_name + '_key_phrase_topics.json')
            key_phrase_groups_df = pd.read_json(path)
            cluster_df['KeywordClusters'] = key_phrase_groups_df['KeywordClusters']
            # Compute the percent
            total = cluster_df['NumDocs'].sum()
            cluster_df['Percent'] = cluster_df['NumDocs'].apply(lambda x: x / total)
            # Output the overall results
            df = cluster_df[['Cluster', 'Score', 'NumDocs', 'Percent', 'DocIds', 'Terms', 'KeywordClusters']]
            # # # # Write to a json file
            folder = os.path.join('output', self.args.case_name, self.args.folder)
            path = os.path.join(folder, self.args.case_name + '_cluster_terms_key_phrases_topics.json')
            df.to_json(path, orient='records')
            # Write to a csv file
            path = os.path.join(folder, self.args.case_name + '_cluster_terms_key_phrases_topics.csv')
            df.to_csv(path, encoding='utf-8', index=False)
            print("Print results to " + path)
        except Exception as err:
            print("Error occurred! {err}".format(err=err))

    # Collect and generate statistics from results
    def collect_statistics(self):
        try:
            folder = os.path.join('output', self.args.case_name, self.args.folder)
            path = os.path.join(folder, self.args.case_name + '_cluster_terms_key_phrases_LDA_topics.json')
            df = pd.read_json(path)
            results = list()
            for index, row in df.iterrows():
                cluster_no = row['Cluster']
                result = {'cluster': cluster_no}
                keyword_clusters = row['KeyPhrases']
                for group_id in range(0, 6):
                    if group_id < len(keyword_clusters):
                        result['keyword_cluster#' + str(group_id + 1)] = len(keyword_clusters[group_id]['DocIds'])
                results.append(result)
            # Write keyword group results to a summary (csv)
            path = os.path.join('output', self.args.case_name, self.args.folder,
                                "keyword_clusters.csv")
            df = pd.DataFrame(results, columns=['cluster', "keyword_cluster#1", "keyword_cluster#2",
                                                "keyword_cluster#3", "keyword_cluster#4", "keyword_cluster#5"])
            df.to_csv(path, encoding='utf-8', index=False)

        except Exception as err:
            print("Error occurred! {err}".format(err=err))


# Main entry
if __name__ == '__main__':
    try:
        # _cluster_no = 2
        # ct = ClusterTopicLDA(_cluster_no)
        ct = TopicKeywordCluster()
        ct.collect_topics_from_keyword_cluster()
        ct.combine_topics_keyword_cluster_to_file()
        # ct.collect_statistics()
    except Exception as err:
        print("Error occurred! {err}".format(err=err))
