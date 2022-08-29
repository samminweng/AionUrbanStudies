import os.path
import sys
from argparse import Namespace
from functools import reduce

import umap
from sentence_transformers import SentenceTransformer
from pathlib import Path
import pandas as pd
import logging
import getpass
# Set logging level
from KeywordGroupUtility import KeywordGroupUtility

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
# Set Sentence Transformer path
sentence_transformers_path = os.path.join('/Scratch', getpass.getuser(), 'SentenceTransformer')
if os.name == 'nt':
    sentence_transformers_path = os.path.join("C:", os.sep, "Users", getpass.getuser(), "SentenceTransformer")
Path(sentence_transformers_path).mkdir(parents=True, exist_ok=True)


# Extract keyword and group keywords based on the similarity
class KeywordGroup:
    def __init__(self):
        self.args = Namespace(
            case_name='AIMLUrbanStudyCorpus',
            embedding_name='OpenAIEmbedding',
            model_name='curie',
            phase='keyword_grouping_phase',
            previous_phase='keyword_extraction_phase',
            path='data',
            diversity=0.5
        )
        # Load corpus dataset
        path = os.path.join('output', self.args.case_name + '_' + self.args.embedding_name, self.args.previous_phase,
                            self.args.case_name + '_clusters.json')
        self.corpus_df = pd.read_json(path)
        self.corpus = self.corpus_df.to_dict("records")
        # Added 'Text' column
        # self.corpus_df['Text'] = self.corpus_df['Title'] + ". " + self.corpus_df['Abstract']
        # Load cluster results
        # Group all docId of a cluster
        path = os.path.join('output', self.args.case_name + '_' + self.args.embedding_name, self.args.previous_phase,
                            self.args.case_name + '_cluster_terms.json')
        cluster_df = pd.read_json(path)
        self.clusters = cluster_df.to_dict("records")

    # Group the key phrases in a cluster with different parameters using HDBSCAN clustering
    def experiment_group_keywords(self):
        folder = os.path.join('output', self.args.case_name + '_' + self.args.embedding_name, self.args.phase)
        # Load keyword vectors
        path = os.path.join('output', self.args.case_name + '_' + self.args.embedding_name, self.args.previous_phase,
                            'keyword_vectors.json')
        keyword_vectors = pd.read_json(path, compression='gzip').to_dict("records")
        # # # Language model
        # model = SentenceTransformer(self.args.model_name, cache_folder=sentence_transformers_path,
        #                             device=self.args.device)
        for cluster in self.clusters:
            try:
                cluster_id = cluster['cluster']
                # if cluster_id < 17:
                #     continue
                key_phrase_folder = os.path.join('output', self.args.case_name + '_' + self.args.embedding_name,
                                                 self.args.previous_phase, 'doc_keywords')
                path = os.path.join(key_phrase_folder, 'doc_keyword_cluster_#' + str(cluster_id) + '.json')
                df = pd.read_json(path)
                # Aggregate the keywords from each individual paper
                cluster_keywords = reduce(lambda pre, cur: pre + cur, df['keywords'].tolist(), list())
                # Filter duplicate key phrases
                cluster_keywords = KeywordGroupUtility.filter_unique_phrases(cluster_keywords)
                # Convert key phrases into vectors
                cluster_keyword_vectors = list()
                for keyword in cluster_keywords:
                    k_vector = next(
                        (kv['vector'] for kv in keyword_vectors if kv['keyword'].lower() == keyword.lower()), None)
                    assert k_vector is not None
                    cluster_keyword_vectors.append(k_vector)
                # Group all keywords using HDBSCAN clustering
                results = KeywordGroupUtility.group_keywords_by_clusters_HDBSCAN(cluster_keywords,
                                                                                 cluster_keyword_vectors)
                # output the experiment results
                df = pd.DataFrame(results)
                experiment_folder = os.path.join(folder, 'experiments')
                Path(experiment_folder).mkdir(parents=True, exist_ok=True)
                path = os.path.join(experiment_folder,
                                    'experiment_keyword_group_cluster#' + str(cluster_id) + '.json')
                df.to_json(path, orient='records')
                path = os.path.join(experiment_folder,
                                    'experiment_keyword_group_cluster#' + str(cluster_id) + '.csv')
                df.to_csv(path, encoding='utf-8', index=False)
                print("=== Complete grouping the key phrases of cluster {no} ===".format(no=cluster_id))
            except Exception as err:
                print("Error occurred! {err}".format(err=err))

    # Obtains the best keyword grouping with best silhouette scores
    def obtain_best_keyword_groupings(self):
        folder = os.path.join('output', self.args.case_name + '_' + self.args.embedding_name, self.args.phase)
        experiment_folder = os.path.join(folder, 'experiments')
        try:
            # Collect the best results in each cluster
            results = list()
            for cluster in self.clusters:
                try:
                    cluster_id = cluster['cluster']
                    # Load the experiment results of the cluster
                    path = os.path.join(experiment_folder,
                                        'experiment_keyword_group_cluster#{c}.json'.format(c=cluster_id))
                    experiments = pd.read_json(path).to_dict("records")
                    # Sort the experiment results by score
                    experiments = sorted(experiments, key=lambda ex: (ex['score'], ex['min_cluster_size']),
                                         reverse=True)
                    # Get the best results
                    best_ex = experiments[0]
                    keyword_group_results = best_ex['group_results']
                    keywords = best_ex['keywords']
                    # x, y position
                    x_pos_list = best_ex['x']
                    y_pos_list = best_ex['y']
                    keyword_positions = list()
                    # combine keyword with its x and y value
                    for keyword, x, y in zip(keywords, x_pos_list, y_pos_list):
                        keyword_positions.append({'keyword': keyword, 'x': x, 'y': y})
                    # Get cluster docs
                    cluster_docs = list(filter(lambda d: d['Cluster'] == cluster_id, self.corpus))
                    # Assign grouped keywords with their x and y positions
                    for keyword_groups in keyword_group_results:
                        grouped_keywords = keyword_groups['keywords']
                        keyword_groups['num_keywords'] = len(grouped_keywords)
                        keyword_groups['x'] = list()
                        keyword_groups['y'] = list()
                        # Collect the docs that contain the keywords in a group
                        group_doc_ids = set()
                        # keyword's projected x and y values
                        for keyword in grouped_keywords:
                            try:
                                found = next(k for k in keyword_positions if k['keyword'].lower() == keyword.lower())
                                keyword_groups['x'].append(found['x'])
                                keyword_groups['y'].append(found['y'])
                                # Get the docs associated with a grouping of keywords
                                for doc in cluster_docs:
                                    doc_id = doc['DocId']
                                    doc_keywords = doc['GPTKeywords']
                                    # Check if doc keywords contains any keywords from grouped keywords
                                    found_keyword = next((k for k in doc_keywords if k.lower() == keyword.lower()),
                                                         None)
                                    if found_keyword:
                                        group_doc_ids.add(doc_id)
                            except Exception as _err:
                                print("Error occurred! {err}".format(err=_err))
                                sys.exit(-1)
                        keyword_groups['doc_ids'] = sorted(list(group_doc_ids))
                        keyword_groups['num_docs'] = len(group_doc_ids)
                        # # Output the results to a chart
                    # print(keyword_group_results)
                    # Output the group results to csv and json files
                    result_folder = os.path.join(folder, 'result')
                    Path(result_folder).mkdir(parents=True, exist_ok=True)
                    # # Output the grouped key phrases
                    group_df = pd.DataFrame(keyword_group_results,
                                            columns=['group', 'score', 'num_keywords', 'keywords',
                                                     'num_docs', 'doc_ids', 'x', 'y'])
                    path = os.path.join(result_folder, 'keyword_groups_cluster_#' + str(cluster_id) + '.csv')
                    group_df.to_csv(path, encoding='utf-8', index=False)
                    path = os.path.join(result_folder, 'keyword_groups_cluster_#' + str(cluster_id) + '.json')
                    group_df.to_json(path, orient='records')
                    print('Output keyword groupings to ' + path)
                    # Store the grouped key phrases of a cluster
                    results.append({'cluster': cluster_id, 'keyword_groups': keyword_group_results})
                    # Output the keyword groups to folder
                    KeywordGroupUtility.visualise_keyword_group_results(cluster_id, keyword_group_results,
                                                                        result_folder)
                except Exception as _err:
                    print("Error occurred! {err}".format(err=_err))
                    sys.exit(-1)
            # # write best results of each group
            df = pd.DataFrame(results,
                              columns=['cluster', 'keyword_groups'])
            path = os.path.join(folder, 'keyword_groups.csv')
            df.to_csv(path, encoding='utf-8', index=False)
            path = os.path.join(folder, 'keyword_groups.json')
            df.to_json(path, orient="records")
        except Exception as err:
            print("Error occurred! {err}".format(err=err))

    # Re-cluster outlier key phrases
    def re_cluster_key_phrases_within_keyword_cluster(self):
        folder = os.path.join('output', self.args.case_name, self.args.cluster_folder, 'key_phrases')
        # Load cluster_key_phrase_groups
        path = os.path.join(folder, 'key_phrase_clusters', 'key_phrases_cluster.json')
        clusters = pd.read_json(path).to_dict("records")
        # minimal cluster size
        cluster_no_list = self.cluster_no_list
        # cluster_no_list = [17]
        try:
            for cluster_no in cluster_no_list:
                is_stop = False
                iteration = 1
                while not is_stop:  # Maximum iteration: 4
                    prev_iteration = iteration - 1
                    try:
                        cluster = next(cluster for cluster in clusters if cluster['Cluster'] == cluster_no)
                        # Load the best grouping of previous level
                        path = os.path.join(folder, 'key_phrase_clusters', 'level_' + str(prev_iteration),
                                            'key_phrases_cluster_#' + str(cluster_no) + '.json')
                        key_phrase_clusters = pd.read_json(path).to_dict("records")
                        # Load doc_key_phrases
                        path = os.path.join(folder, 'doc_key_phrase',
                                            'doc_key_phrases_cluster_#' + str(cluster_no) + ".json")
                        doc_key_phrases = pd.read_json(path).to_dict("records")
                        # Load key phrase vectors
                        path = os.path.join(folder, 'key_phrase_vectors',
                                            'key_phrase_vectors_cluster#' + str(cluster_no) + '.json')
                        key_phrase_vectors = pd.read_json(path).to_dict("records")
                        # # print(doc_key_phrases)
                        # Re-cluster keyword cluster
                        updated_key_phrase_clusters = KeywordGroupUtility.run_re_clustering_experiments(cluster_no,
                                                                                                        key_phrase_clusters,
                                                                                                        key_phrase_vectors,
                                                                                                        doc_key_phrases)

                        cluster['Key-phrases'] = updated_key_phrase_clusters
                        out_folder = os.path.join(folder, 'key_phrase_clusters', 'level_' + str(iteration))
                        Path(out_folder).mkdir(parents=True, exist_ok=True)
                        # Write new key phrase groups to
                        df = pd.DataFrame(updated_key_phrase_clusters)
                        df = df[['Group', 'score', 'NumPhrases', 'Key-phrases', 'NumDocs', 'DocIds', 'dimension',
                                 'min_samples', 'min_cluster_size', 'x', 'y']]
                        path = os.path.join(out_folder,
                                            'key_phrases_cluster_#' + str(cluster_no) + '.csv')
                        df.to_csv(path, encoding='utf-8', index=False)
                        path = os.path.join(out_folder,
                                            'key_phrases_cluster_#' + str(cluster_no) + '.json')
                        df.to_json(path, orient="records")
                        KeywordGroupUtility.visualise_keywords_cluster_results(cluster_no,
                                                                               updated_key_phrase_clusters,
                                                                               out_folder)
                        print("=== Complete re-clustering key phrases in cluster #{c_no} at Iteration {i} ===".format(
                            c_no=cluster_no, i=iteration))
                        is_stop = KeywordGroupUtility.check_stop_iteration(updated_key_phrase_clusters,
                                                                           len(doc_key_phrases))
                        iteration = iteration + 1
                    except Exception as _err:
                        print("Error occurred! {err}".format(err=_err))
                        sys.exit(-1)
            # # Write output to 'cluster_key_phrases_group'
            df = pd.DataFrame(clusters,
                              columns=['Cluster', 'Key-phrases'])
            folder = os.path.join('output', self.args.case_name, self.args.cluster_folder,
                                  'key_phrases', 'key_phrase_clusters')
            Path(folder).mkdir(parents=True, exist_ok=True)
            path = os.path.join(folder, 'key_phrases_cluster.csv')
            df.to_csv(path, encoding='utf-8', index=False)
            path = os.path.join(folder, 'key_phrases_cluster.json')
            df.to_json(path, orient="records")
        except Exception as _err:
            print("Error occurred! {err}".format(err=_err))
            sys.exit(-1)

    # Combine the cluster terms and keyword group results
    def combine_cluster_terms_keyword_groups(self):
        try:
            folder = os.path.join('output', self.args.case_name + '_' + self.args.embedding_name, self.args.phase)
            # Combine all key phrases and TF-IDF topics to a json file
            # Load cluster terms
            path = os.path.join('output', self.args.case_name + '_' + self.args.embedding_name,
                                self.args.previous_phase,
                                self.args.case_name + '_cluster_terms.json')
            cluster_df = pd.read_json(path)
            # Load keyword groups
            path = os.path.join(folder, 'keyword_groups.json')
            key_phrase_df = pd.read_json(path)
            cluster_df['keyword_groups'] = key_phrase_df['keyword_groups'].tolist()
            # rename 'group' as cluster group
            cluster_df['cluster_group'] = cluster_df['group'].tolist()
            # Re-order cluster df and Output to csv and json file
            cluster_df = cluster_df[['iteration', 'cluster_group', 'cluster', 'score', 'count', 'doc_ids',
                                     'freq_terms', 'keyword_groups']]
            path = os.path.join(folder, self.args.case_name + '_cluster_terms_keyword_groups.csv')
            cluster_df.to_csv(path, encoding='utf-8', index=False)
            path = os.path.join(folder, self.args.case_name + '_cluster_terms_keyword_groups.json')
            cluster_df.to_json(path, orient='records')
            print('Output keyword groups per cluster to ' + path)

        except Exception as err:
            print("Error occurred! {err}".format(err=err))

    # Find the common terms that appear frequently (more than two abstracts) in a cluster
    def obtain_common_terms_by_clusters(self):
        # Collect the common terms from each cluster's frequent terms
        def collect_common_terms_from_cluster_freq_terms(_grouped_clusters):
            try:
                # Collect the common terms
                _common_terms_dict = {}
                # Collect the frequencies of each term that appear in the clusters
                for _cluster in _grouped_clusters:
                    _cluster_freq_terms = _cluster['unique_terms']
                    for _term in _cluster_freq_terms:
                        if _term['term'].lower() not in _common_terms_dict:
                            _common_terms_dict.setdefault(_term['term'].lower(), 0)
                        _common_terms_dict[_term['term'].lower()] = _common_terms_dict[_term['term'].lower()] + 1
                # Filter out common terms
                _common_terms = list(_term for _term, _freq in _common_terms_dict.items() if _freq > 1)
                # print(_common_terms)
                return _common_terms
            except Exception as _e:
                print("Error occurred! {err}".format(err=_e))

        try:
            folder = os.path.join('output', self.args.case_name + '_' + self.args.embedding_name, self.args.phase)
            path = os.path.join(folder, self.args.case_name + '_cluster_terms_keyword_groups.json')
            cluster_df = pd.read_json(path)
            clusters = cluster_df.to_dict("records")
            # Get the maximal cluster group
            max_group = cluster_df['cluster_group'].max()
            results = list()
            # Go through each cluster group
            for cluster_group_no in range(0, max_group + 1):
                try:
                    iteration = 1
                    common_terms = list()
                    # Get the clusters within the same cluster group
                    grouped_clusters = list(filter(lambda c: c['cluster_group'] == cluster_group_no, clusters))
                    # Initialize the cluster's updated_freq_terms
                    for cluster in grouped_clusters:
                        cluster['unique_terms'] = cluster['freq_terms'][:10]
                    # Get the common terms
                    common_terms = common_terms + collect_common_terms_from_cluster_freq_terms(grouped_clusters)
                    # Removed duplicated
                    common_terms = list(dict.fromkeys(common_terms))
                    is_full = False
                    while not is_full:
                        is_full = True
                        # filter common terms from each cluster
                        for index, cluster in enumerate(grouped_clusters):
                            cluster_selected_terms = cluster['unique_terms']
                            updated_cluster_terms = list(filter(lambda t: t['term'].lower() not in common_terms, cluster_selected_terms))
                            if len(updated_cluster_terms) < 10:
                                diff = 10 - len(updated_cluster_terms)
                                terms = list(map(lambda t: t['term'].lower(), updated_cluster_terms))
                                # Get the freq terms that do not contain common terms or updated_cluster_terms
                                remaining_terms = list(filter(lambda t: t['term'].lower() not in common_terms and
                                                                        t['term'].lower() not in terms,
                                                              cluster['freq_terms']))
                                updated_cluster_terms = updated_cluster_terms + remaining_terms[:diff]
                                is_full = is_full & False
                            else:
                                is_full = True
                            # Each cluster has 10 terms
                            grouped_clusters[index]['unique_terms'] = updated_cluster_terms
                            # print(grouped_clusters)
                        # Start another iteration
                        print("Start iteration {d} to find the common terms {a}".format(d=iteration,a=common_terms))
                        iteration = iteration + 1
                        # # Stop searching common terms
                        if is_full:
                            # Update the cluster's common terms
                            for cluster in grouped_clusters:
                                cluster['common_terms'] = common_terms
                                results.append(cluster)
                            print("Complete finding common terms for cluster group {g}".format(g=cluster_group_no))
                except Exception as e:
                    print("Error occurred! {err}".format(err=e))
                    sys.exit(-1)
            print(results)
            folder = os.path.join('output', self.args.case_name + '_' + self.args.embedding_name)
            df = pd.DataFrame(results, columns=['iteration', 'cluster_group', 'cluster', 'score', 'count', 'doc_ids',
                                                'keyword_groups', 'freq_terms', 'unique_terms', 'common_terms'])
            path = os.path.join(folder, self.args.case_name + '_cluster_terms_keyword_groups.csv')
            df.to_csv(path, encoding='utf-8', index=False)
            # Write article corpus to a json file
            path = os.path.join(folder,
                                self.args.case_name + '_cluster_terms_keyword_groups.json')
            df.to_json(path, orient='records')
            # Output the corpus
            path = os.path.join(folder, self.args.case_name + '_clusters.csv')
            self.corpus_df.to_csv(path, encoding='utf-8', index=False)
            path = os.path.join(folder, self.args.case_name + '_clusters.json')
            self.corpus_df.to_json(path, orient='records')
            print('Output docs in the corpus to ' + path)
        except Exception as e:
            print("Error occurred! {err}".format(err=e))


# Main entry
if __name__ == '__main__':
    try:
        kp = KeywordGroup()
        # Run the keyword grouping experiments
        # kp.experiment_group_keywords()
        # kp.obtain_best_keyword_groupings()
        # kp.combine_cluster_terms_keyword_groups()
        kp.obtain_common_terms_by_clusters()
    except Exception as err:
        print("Error occurred! {err}".format(err=err))
