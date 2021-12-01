import os.path
from argparse import Namespace
from sentence_transformers import SentenceTransformer
from pathlib import Path
import pandas as pd
import logging
from KeyPhraseUtility import KeyPhraseUtility
import getpass

# Set logging level
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
# Set Sentence Transformer path
sentence_transformers_path = os.path.join('/Scratch', getpass.getuser(), 'SentenceTransformer')
if os.name == 'nt':
    sentence_transformers_path = os.path.join("C:", os.sep, "Users", getpass.getuser(), "SentenceTransformer")
Path(sentence_transformers_path).mkdir(parents=True, exist_ok=True)


# Convert the cluster article title into a vector and find the similar articles in other cluster
class ClusterSimilarity:
    def __init__(self):
        self.args = Namespace(
            case_name='UrbanStudyCorpus',
            approach='HDBSCAN',
            # Model name ref: https://www.sbert.net/docs/pretrained_models.html
            model_name="all-mpnet-base-v2",
            device='cuda'
        )
        # Load the cluster results as dataframe
        path = os.path.join('output', 'cluster', self.args.case_name + "_HDBSCAN_Cluster_TF-IDF_topic_words.json")
        df = pd.read_json(path)
        self.clusters = df.to_dict("records")  # Convert to a list of dictionaries
        # Load the published paper text data
        path = os.path.join('data', self.args.case_name + '.json')
        corpus_df = pd.read_json(path)
        corpus_df['Text'] = corpus_df['Title'] + ". " + corpus_df['Abstract']
        # Load HDBSCAN cluster
        path = os.path.join('output', 'cluster', self.args.case_name + "_clusters.json")
        hdbscan_cluster_df = pd.read_json(path)
        # Update corpus data with hdbscan cluster results
        corpus_df['Cluster'] = hdbscan_cluster_df['HDBSCAN_Cluster']
        self.corpus_docs = corpus_df.to_dict("records")
        duplicate_doc_ids = KeyPhraseUtility.scan_duplicate_articles()
        # Remove duplicated doc
        self.corpus_docs = list(filter(lambda d: d['DocId'] not in duplicate_doc_ids, self.corpus_docs))

    # # Use the BERT model to extract long key phrases
    # # Ref: https://towardsdatascience.com/keyword-extraction-with-bert-724efca412ea
    def extract_key_phrases_by_clusters(self):
        # Get a list of unique key phrases from all papers
        def _get_unique_doc_key_phrases(_doc_key_phrases, _all_key_phrases, _top_k=5):
            try:
                if len(_doc_key_phrases) < _top_k:
                    return _doc_key_phrases

                _unique_key_phrases = list()
                for _key_phrase in _doc_key_phrases:
                    # find if key phrase exist in all key phrase list
                    _found_key_phrases = [_kp for _kp in _all_key_phrases
                                          if _kp['key-phrase'].lower() == _key_phrase['key-phrase'].lower()]
                    if len(_found_key_phrases) == 0:
                        _unique_key_phrases.append(_key_phrase)
                    else:
                        print("Duplicated: " + _found_key_phrases[0]['key-phrase'])

                # Get top 5 key phrase
                _unique_key_phrases = _unique_key_phrases[:_top_k]
                # assert len(_unique_key_phrases) == _top_k
                return _unique_key_phrases
            except Exception as _err:
                print("Error occurred! {err}".format(err=_err))

        # Write the key phrases of each cluster to a csv file
        def _write_key_phrases_by_cluster(_key_phrase_list):
            try:
                df = pd.DataFrame(_key_phrase_list, columns=['DocId'])
                df['No'] = range(1, len(df)+1)
                # Map the list of key phrases (dict) to a list of strings
                # Map the nested dict to a list of key phrases (string only)
                df['key-phrases'] = list(map(lambda k: [kp['key-phrase'] for kp in k['key-phrases']], _key_phrase_list))
                df = df[['No', 'DocId', 'key-phrases']]     # Re-order the columns
                folder = os.path.join('output', 'key_phrases', 'cluster')
                Path(folder).mkdir(parents=True, exist_ok=True)
                _path = os.path.join(folder, 'top_key_phrases_cluster_#' + str(cluster_no) + '.csv')
                df.to_csv(_path, encoding='utf-8', index=False)
                _path = os.path.join(folder, 'top_key_phrases_cluster_#' + str(cluster_no) + '.json')
                df.to_json(_path, orient='records')
            except Exception as _err:
                print("Error occurred! {err}".format(err=_err))

        try:
            # # Encode cluster_doc and candidates as BERT embedding
            model = SentenceTransformer(self.args.model_name, cache_folder=sentence_transformers_path,
                                        device=self.args.device)
            cluster_no_list = list(range(-1, 10))
            for cluster_no in cluster_no_list:
                cluster_docs = list(filter(lambda d: d['Cluster'] == cluster_no, self.corpus_docs))
                results = list()
                all_key_phrases = list()    # Store all the key phrases
                for doc in cluster_docs:
                    try:
                        doc_id = doc['DocId']
                        # Get the first doc
                        doc = next(doc for doc in cluster_docs if doc['DocId'] == doc_id)
                        sentences = KeyPhraseUtility.clean_sentence(doc['Text'])
                        doc_text = " ".join(list(map(lambda s: " ".join(s), sentences)))
                        result = {'Cluster': cluster_no, 'DocId': doc_id}
                        # Collect all the key phrases of a doc
                        candidates = []
                        for n_gram_range in [1, 2, 3]:
                            try:
                                # Extract key phrase candidates using n-gram
                                n_gram_candidates = KeyPhraseUtility.generate_n_gram_candidates(sentences,
                                                                                                n_gram_range)
                                # find and collect top 30 key phrases similar to a paper
                                top_n_gram_key_phrases = KeyPhraseUtility.collect_top_key_phrases(model, doc_text,
                                                                                                  n_gram_candidates,
                                                                                                  top_k=30)
                                result[str(n_gram_range) + '-gram-key-phrases'] = top_n_gram_key_phrases
                                candidates = candidates + list(map(lambda p: p['key-phrase'], top_n_gram_key_phrases))
                            except Exception as err:
                                print("Error occurred! {err}".format(err=err))
                        # Combine all the n-gram key phrases in a doc
                        # Get top 5 key phrase unique to all key phrase list
                        top_doc_key_phrases = _get_unique_doc_key_phrases(
                                            KeyPhraseUtility.collect_top_key_phrases(model, doc_text, candidates, top_k=30),
                                            all_key_phrases, _top_k=5)
                        # Write top five key phrases to 'doc_key_phrases'
                        result['key-phrases'] = top_doc_key_phrases
                        all_key_phrases = all_key_phrases + top_doc_key_phrases      # Concatenate all key phrases of a doc
                        results.append(result)
                    except Exception as err:
                        print("Error occurred! {err}".format(err=err))
                # Write key phrases to csv file
                _write_key_phrases_by_cluster(results)
                # Cluster all key phrases by using HDBSCAN
                KeyPhraseUtility.cluster_key_phrases_by_HDBSCAN(all_key_phrases, cluster_no, model, is_experimented=True)
        except Exception as err:
            print("Error occurred! {err}".format(err=err))

    # Combine all the key phrases results
    def combine_key_phrases(self):
        # List top 10 key phrase of each group
        def summary_group_key_phrases(_group_key_phrases):
            _group_key_phrases = sorted(_group_key_phrases, key= lambda _g: _g['group'], reverse=True)
            summary = list()
            for _no, _group in enumerate(_group_key_phrases):
                top_key_phrases = _group['key-phrase'].split(", ")[:10]
                g_summary = "({no})\t{s}".format(no=_no+1, s=", ".join(top_key_phrases))
                summary.append(g_summary)
            return "\n".join(summary)

        try:
            # Output key phrases of each paper
            in_folder = os.path.join('output', 'key_phrases', 'cluster')
            out_folder = os.path.join('output', 'key_phrases')
            # Combine all the key phrases of each paper to a json file
            key_phrases = list()
            for cluster_no in list(range(-1, 10)):
                path = os.path.join(in_folder, 'top_key_phrases_cluster_#{c}.json'.format(c=cluster_no))
                df = pd.read_json(path, orient='records')
                doc_key_phrases = df.to_dict("records")
                key_phrases = key_phrases + doc_key_phrases
            # Sort key phrases by DocId
            sorted_key_phrases = sorted(key_phrases, key=lambda k: k['DocId'])
            # Aggregated all the key phrases of each individual article
            df = pd.DataFrame(sorted_key_phrases, columns=['DocId', 'key-phrases'])
            path = os.path.join(out_folder, self.args.case_name + '_doc_key_phrases.csv')
            df.to_csv(path, index=False, encoding='utf-8')
            path = os.path.join(out_folder, self.args.case_name + '_doc_key_phrases.json')
            df.to_json(path, orient='records')
            print('Output key phrases per doc to ' + path)

            # Combine all the key phrases of each cluster and TF-IDF topics to a json file
            # Load TF-IDF topics
            path = os.path.join('output', 'cluster', self.args.case_name + '_' + self.args.approach +
                                '_Cluster_TF-IDF_topic_words.json')
            df = pd.read_json(path)
            clusters = df.to_dict("records")
            # Combine all best grouped key phrases of each cluster
            for cluster_no in list(range(-1, 10)):
                try:
                    path = os.path.join(in_folder, 'top_key_phrases_cluster_#{c}_best_grouping.json'.format(c=cluster_no))
                    df = pd.read_json(path, orient='records')
                    grouped_key_phrases = df.to_dict("records")
                    cluster = next(cluster for cluster in clusters if cluster['Cluster'] == cluster_no)
                    cluster['Grouped_Key_Phrases'] = grouped_key_phrases
                except Exception as err:
                    print("Error occurred! {err}".format(err=err))
            # Output to csv and json file
            cluster_df = pd.DataFrame(clusters, columns=['Cluster', 'NumDocs', 'DocIds',
                                                         'TF-IDF-Topics', 'Grouped_Key_Phrases'])
            path = os.path.join(out_folder, self.args.case_name + '_' + self.args.approach + '_Cluster_topic_key_phrases.csv')
            cluster_df.to_csv(path, encoding='utf-8', index=False)
            path = os.path.join(out_folder,
                                self.args.case_name + '_' + self.args.approach + '_Cluster_topic_key_phrases.json')
            cluster_df.to_json(path, orient='records')
            print('Output key phrases per cluster to ' + path)
            # Output a summary of top 10 Topics and grouped key phrases of each cluster
            clusters = cluster_df.to_dict("records")
            summary_df = cluster_df
            total = summary_df['NumDocs'].sum()
            summary_df['percent'] = list(map(lambda c: c['NumDocs']/total, clusters))
            summary_df['topics'] = list(map(lambda c: ", ".join(list(map(lambda t: t['topic'], c['TF-IDF-Topics'][:10]))), clusters))
            summary_df['key-phrases'] = list(map(lambda c: summary_group_key_phrases(c['Grouped_Key_Phrases']), clusters))
            path = os.path.join(out_folder,
                                self.args.case_name + '_' + self.args.approach + '_Cluster_topic_key_phrases_summary.csv')
            summary_df = summary_df.drop(columns=['TF-IDF-Topics', 'DocIds', 'Grouped_Key_Phrases'])
            summary_df.to_csv(path, encoding='utf-8', index=False)
            print('Output summary of topics and key phrases per cluster to ' + path)
        except Exception as err:
            print("Error occurred! {err}".format(err=err))


    # Find top 30 similar papers for each article in a cluster
    def find_top_similar_paper_in_corpus(self, top_k=30):
        # cluster_no_list = [c_no for c_no in range(-1, 23)]
        cluster_no_list = [-1]
        try:
            model = SentenceTransformer(self.args.model_name, cache_folder=sentence_transformers_path,
                                        device=self.args.device)  # Load sentence transformer model
            # # Find top 30 similar papers of each paper in a cluster
            for cluster_no in cluster_no_list:
                KeyPhraseUtility.find_top_n_similar_papers(cluster_no, self.corpus_docs, self.clusters, model,
                                                           top_k=top_k)
                # # # Summarize the similar paper results
                KeyPhraseUtility.write_to_title_csv_file(cluster_no, top_k=top_k)
        except Exception as err:
            print("Error occurred! {err}".format(err=err))


# Main entry
if __name__ == '__main__':
    tw = ClusterSimilarity()
    # tw.find_top_similar_paper_in_corpus()
    # tw.extract_key_phrases_by_clusters()
    tw.combine_key_phrases()




