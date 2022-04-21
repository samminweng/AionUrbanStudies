import os
import sys
from argparse import Namespace
import getpass
from pathlib import Path
from sentence_transformers import SentenceTransformer
import pandas as pd
from BERTArticleClusterUtility import BERTArticleClusterUtility
from KeyWordExtractionUtility import KeywordExtractionUtility
from stanza.server import CoreNLPClient

# Set Sentence Transformer path
sentence_transformers_path = os.path.join('/Scratch', getpass.getuser(), 'SentenceTransformer')
if os.name == 'nt':
    sentence_transformers_path = os.path.join("C:", os.sep, "Users", getpass.getuser(), "SentenceTransformer")
Path(sentence_transformers_path).mkdir(parents=True, exist_ok=True)


class KeywordExtraction:
    # def __init__(self, _cluster_no):
    def __init__(self):
        self.args = Namespace(
            case_name='AIMLUrbanStudyCorpus',
            # Model name ref: https://www.sbert.net/docs/pretrained_models.html
            model_name="all-mpnet-base-v2",
            device='cpu',
            diversity=0.5,
            cluster_folder='cluster_merge',
        )
        # # Use the BERT model to find top 5 similar key phrases of each paper
        # # Ref: https://towardsdatascience.com/keyword-extraction-with-bert-724efca412ea
        # Load HDBSCAN cluster
        path = os.path.join('output', self.args.case_name, self.args.cluster_folder,
                            self.args.case_name + '_clusters.json')
        self.corpus_df = pd.read_json(path)
        # Update corpus data with hdbscan cluster results
        self.corpus_df.rename(columns={'HDBSCAN_Cluster': 'Cluster'}, inplace=True)
        # Added 'Text' column
        self.corpus_df['Text'] = self.corpus_df['Title'] + ". " + self.corpus_df['Abstract']
        # Get the total cluster
        self.cluster_no_list = sorted(list(dict.fromkeys(self.corpus_df['Cluster'].tolist())))
        # self.cluster_no_list = [3, 8]
        # Group all docId of a cluster
        cluster_df = self.corpus_df.groupby(['Cluster'], as_index=False).agg(
            {'DocId': lambda doc_id: list(doc_id), 'Text': lambda text: list(text)})
        cluster_df.rename(columns={'DocId': 'DocIds'}, inplace=True)
        cluster_df['NumDocs'] = cluster_df['DocIds'].apply(len)
        cluster_df = cluster_df[['Cluster', 'NumDocs', 'DocIds']]
        self.clusters = cluster_df.to_dict("records")
        # # Language model
        self.model = SentenceTransformer(self.args.model_name, cache_folder=sentence_transformers_path,
                                         device=self.args.device)

    def extract_doc_key_phrases_by_similarity_diversity(self):
        try:
            folder = os.path.join('output', self.args.case_name, self.args.cluster_folder,
                                  'key_phrases', 'doc_key_phrase')
            Path(folder).mkdir(parents=True, exist_ok=True)
            corpus_docs = self.corpus_df.to_dict("records")
            # Collect all the tfidf terms from all docs
            # # A folder that stores all the topic results
            tfidf_folder = os.path.join(folder, 'tf-idf')
            Path(tfidf_folder).mkdir(parents=True, exist_ok=True)
            # Extract single-word candidates using TF-IDF
            tfidf_candidates = KeywordExtractionUtility.generate_tfidf_terms(corpus_docs, tfidf_folder)
            # Collect collocation from each cluster of articles
            with CoreNLPClient(
                    annotators=['tokenize', 'ssplit', 'pos'],
                    timeout=30000,
                    memory='6G') as client:
                # cluster_no_list = [8]
                cluster_no_list = self.cluster_no_list
                for cluster_no in cluster_no_list:
                    cluster_docs = list(filter(lambda d: d['Cluster'] == cluster_no, corpus_docs))
                    results = list()  # Store all the key phrases
                    for doc in cluster_docs:
                        doc_id = doc['DocId']
                        # if doc_id != 523:
                        #     continue
                        # Get the first doc
                        doc = next(doc for doc in cluster_docs if doc['DocId'] == doc_id)
                        doc_text = BERTArticleClusterUtility.preprocess_text(doc['Abstract'])
                        # End of for loop
                        try:
                            doc_tfidf_candidates = next(c for c in tfidf_candidates if c['doc_id'] == doc_id)['terms']
                            # Get top 2 uni_grams from tf-idf terms
                            uni_gram_candidates = doc_tfidf_candidates[:2]
                            # Collect all the candidate collocation words
                            n_gram_candidates = KeywordExtractionUtility.generate_collocation_candidates(doc_text,
                                                                                                         client)
                            n_gram_candidates = n_gram_candidates + list(map(lambda c: c['term'], uni_gram_candidates))
                            # print(", ".join(n_gram_candidates))
                            candidate_scores = KeywordExtractionUtility.compute_similar_score_key_phrases(self.model,
                                                                                                          doc_text,
                                                                                                          n_gram_candidates)

                            phrase_similar_scores = KeywordExtractionUtility.sort_phrases_by_similar_score(
                                candidate_scores)
                            phrase_candidates = list(map(lambda p: p['key-phrase'], phrase_similar_scores))
                            # Rank the high scoring phrases
                            phrase_scores_mmr = KeywordExtractionUtility.re_rank_phrases_by_maximal_margin_relevance(
                                self.model, doc_text, phrase_candidates, self.args.diversity)
                            mmr_key_phrases = list(map(lambda p: p['key-phrase'], phrase_scores_mmr))
                            # filter out single word overlapping with any other
                            top_key_phrases = list()
                            for key_phrase in mmr_key_phrases:
                                if len(key_phrase.split(" ")) == 1:
                                    single_word = key_phrase.lower()
                                    # Check if the single word overlaps with existing words
                                    found = next(
                                        (phrase for phrase in top_key_phrases if single_word != phrase.lower() and
                                         single_word in phrase.lower()), None)
                                    if not found:
                                        top_key_phrases.append(key_phrase)
                                else:
                                    top_key_phrases.append(key_phrase)

                            # Obtain top five key phrases
                            result = {'Cluster': cluster_no, 'DocId': doc_id,
                                      'Key-phrases': top_key_phrases[:5],
                                      'Candidate-count': len(phrase_similar_scores),
                                      'Phrase-candidates': phrase_similar_scores}
                            results.append(result)
                            print("Complete to extract the key phrases from document {d_id}".format(d_id=doc_id))
                        except Exception as __err:
                            print("Error occurred! {err}".format(err=__err))
                            sys.exit(-1)
                    print(results)
                    # Write key phrases to csv file
                    df = pd.DataFrame(results)
                    # Map the list of key phrases (dict) to a list of strings
                    Path(folder).mkdir(parents=True, exist_ok=True)
                    path = os.path.join(folder, 'doc_key_phrases_cluster_#' + str(cluster_no) + '.csv')
                    df.to_csv(path, encoding='utf-8', index=False)
                    path = os.path.join(folder, 'doc_key_phrases_cluster_#' + str(cluster_no) + '.json')
                    df.to_json(path, orient='records')
                    print("Output the key phrases of cluster #" + str(cluster_no))
        except Exception as err:
            print("Error occurred! {err}".format(err=err))

    # Combine clusters and doc key phrases
    def combine_cluster_doc_key_phrases(self):
        # Combine all the doc key phrases into a single file 'doc_key_phrases'
        try:
            key_phrase_folder = os.path.join('output', self.args.case_name, self.args.cluster_folder, 'key_phrases')
            # Combine the key phrases of all papers to a single file
            doc_key_phrases = list()
            for cluster_no in self.cluster_no_list:
                # Get key phrases of a cluster
                path = os.path.join(key_phrase_folder, 'doc_key_phrase',
                                    'doc_key_phrases_cluster_#{c}.json'.format(c=cluster_no))
                docs = pd.read_json(path).to_dict("records")
                for doc in docs:
                    doc_key_phrases.append({'DocId': doc['DocId'], 'KeyPhrases': doc['Key-phrases'],
                                           'CandidatePhrases': doc['Phrase-candidates']})
            # Sort key phrases by DocId
            sorted_key_phrases = sorted(doc_key_phrases, key=lambda k: k['DocId'])
            # # Aggregated all the key phrases of each individual article
            df = pd.DataFrame(sorted_key_phrases)
            # Combine cluster and doc key phrases
            self.corpus_df['KeyPhrases'] = df['KeyPhrases'].tolist()
            self.corpus_df['CandidatePhrases'] = df['CandidatePhrases'].tolist()
            # Drop column
            self.corpus_df = self.corpus_df.drop('Text', axis=1)
            folder = os.path.join('output', self.args.case_name, self.args.cluster_folder)
            path = os.path.join(folder, self.args.case_name + '_clusters.csv')
            self.corpus_df.to_csv(path, index=False, encoding='utf-8')
            path = os.path.join(folder, self.args.case_name + '_clusters.json')
            self.corpus_df.to_json(path, orient='records')
            print('Output key phrases per doc to ' + path)
        except Exception as err:
            print("Error occurred! {err}".format(err=err))


# Main entry
if __name__ == '__main__':
    try:
        kp = KeywordExtraction()
        # Extract keyword for each article
        kp.extract_doc_key_phrases_by_similarity_diversity()
        kp.combine_cluster_doc_key_phrases()
    except Exception as err:
        print("Error occurred! {err}".format(err=err))
