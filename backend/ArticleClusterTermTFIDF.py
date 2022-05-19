import getpass
import os
import sys
from argparse import Namespace
from functools import reduce
from pathlib import Path

import numpy as np
import pandas as pd
# Obtain the cluster results of the best results and extract cluster topics using TF-IDF
from sentence_transformers import SentenceTransformer

from ArticleClusterTermTFIDFUtility import ArticleClusterTermTFIDFUtility

# Set Sentence Transformer path
sentence_transformers_path = os.path.join('/Scratch', getpass.getuser(), 'SentenceTransformer')
if os.name == 'nt':
    sentence_transformers_path = os.path.join("C:", os.sep, "Users", getpass.getuser(), "SentenceTransformer")
Path(sentence_transformers_path).mkdir(parents=True, exist_ok=True)


class ArticleClusterTermTFIDF:
    def __init__(self):
        self.args = Namespace(
            case_name='AIMLUrbanStudyCorpus',
            approach='TF-IDF',
            cluster_folder='cluster_merge',
            model_name='all-mpnet-base-v2',
            device='cpu',
        )
        # Load Sentence Transformer
        self.model = SentenceTransformer(self.args.model_name, cache_folder=sentence_transformers_path,
                                         device=self.args.device)

    # def __init__(self, _last_iteration, _cluster_no):
    #     self.args = Namespace(
    #         case_name='AIMLUrbanStudyCorpus',
    #         approach='TF-IDF',
    #         cluster_folder='cluster_' + str(_cluster_no),
    #         # cluster_no=_cluster_no,
    #         in_folder='iteration',
    #         last_iteration=_last_iteration
    #     )
    # Collect all iterative cluster results
    def collect_iterative_cluster_results(self):
        cluster_folder = os.path.join('output', self.args.case_name, self.args.cluster_folder, 'cluster')
        results = list()
        # Go through each iteration 1 to last iteration
        for i in range(0, self.args.last_iteration + 1):
            try:
                opt_dimension = 0
                # Get the best dimension
                folder = os.path.join(cluster_folder, self.args.in_folder + '_' + str(i))
                clustering_folder = os.path.join(folder, 'hdbscan_clustering')
                # Get the optimal dimension
                for file in os.listdir(clustering_folder):
                    file_name = file.lower()
                    if file_name.endswith(".png") and file_name.startswith("dimension"):
                        opt_dimension = int(file_name.split("_")[1].split(".png")[0])
                # Get the best score of all clustering experiments
                ex_folder = os.path.join(folder, 'experiments')
                path = os.path.join(ex_folder, 'HDBSCAN_cluster_doc_vector_result_summary.json')
                experiment_results = pd.read_json(path).to_dict("records")
                best_result = next(r for r in experiment_results if r['dimension'] == opt_dimension)
                min_samples = best_result['min_samples']
                min_cluster_size = best_result['min_cluster_size']
                score = best_result['Silhouette_score']
                # Get summary of cluster topics
                path = os.path.join(folder, self.args.case_name + '_cluster_docs.json')
                clusters = pd.read_json(path).to_dict("records")
                total_papers = reduce(lambda total, ct: ct['NumDocs'] + total, clusters, 0)
                for cluster in clusters:
                    results.append({
                        "iteration": i, "total_papers": total_papers, "dimension": opt_dimension,
                        "min_samples": min_samples, "min_cluster_size": min_cluster_size, "score": score,
                        "Cluster": cluster['HDBSCAN_Cluster'], "NumDocs": cluster['NumDocs'],
                        "Percent": cluster['Percent'],
                        "DocIds": cluster['DocIds']
                    })
            except Exception as _err:
                print("Error occurred! {err}".format(err=_err))
        # Load the results as data frame
        df = pd.DataFrame(results)
        # Output cluster results to CSV
        folder = os.path.join('output', self.args.case_name, self.args.cluster_folder, 'cluster_terms',
                              'iterative_clusters')
        Path(folder).mkdir(parents=True, exist_ok=True)
        path = os.path.join(folder, self.args.case_name + '_iterative_summary.csv')
        df.to_csv(path, encoding='utf-8', index=False)
        path = os.path.join(folder, self.args.case_name + '_iterative_summary.json')
        df.to_json(path, orient='records')
        print(df)

    # Collect all the iterative cluster results and combine into a single cluster results
    # Output the iterative cluster results
    def output_iterative_cluster_results(self):
        score_path = os.path.join('output', self.args.case_name, self.args.cluster_folder, 'cluster_terms',
                                  'iterative_clusters', self.args.case_name + '_iterative_summary.json')
        scores = pd.read_json(score_path).to_dict("records")
        # Load cluster results at 0 iteration as initial state
        cur_cluster_no = -1
        results = list()
        # Go through each iteration 1 to last iteration
        for iteration in range(0, self.args.last_iteration + 1):
            try:
                iter_score = list(filter(lambda s: s['iteration'] == iteration, scores))
                score = iter_score[0]['score']
                folder = os.path.join('output', self.args.case_name, self.args.cluster_folder, 'cluster')
                # Load the clustered docs in each iteration
                cluster_path = os.path.join(folder, self.args.in_folder + '_' + str(iteration),
                                            self.args.case_name + '_clusters.json')
                df = pd.read_json(cluster_path)
                df['Score'] = score
                cluster_df = df
                total_cluster_no = cluster_df['HDBSCAN_Cluster'].max()
                cluster_no_list = list(range(-1, total_cluster_no + 1))
                # Added the clustered results
                for cluster_no in cluster_no_list:
                    # Get the clustered docs
                    c_df = cluster_df[cluster_df['HDBSCAN_Cluster'] == cluster_no]
                    docs = c_df.to_dict("records")
                    if len(docs) < 40:
                        for doc in docs:
                            doc['HDBSCAN_Cluster'] = cur_cluster_no
                        results.extend(docs)
                        cur_cluster_no = cur_cluster_no + 1
                copied_results = results.copy()
                image_folder = os.path.join('output', self.args.case_name, self.args.cluster_folder,
                                            'cluster_terms', 'images')
                Path(image_folder).mkdir(parents=True, exist_ok=True)
                file_path = os.path.join(image_folder, 'iteration_' + str(iteration) + ".png")
                title = 'Iteration = ' + str(iteration)
                # Visualise the cluster results
                ArticleClusterTermTFIDFUtility.visualise_cluster_results_by_iteration(title, copied_results, file_path)
            except Exception as _err:
                print("Error occurred! {err}".format(err=_err))
        # # Sort the results by DocID
        results = sorted(results, key=lambda c: c['DocId'])
        text_df = pd.DataFrame(results)
        # Reorder the columns
        text_df = text_df[['HDBSCAN_Cluster', 'Score', 'DocId', 'Cited by', 'Year', 'Document Type',
                           'Title', 'Abstract', 'Author Keywords', 'Authors', 'DOI', 'x', 'y']]
        # Output cluster results to CSV
        folder = os.path.join('output', self.args.case_name, self.args.cluster_folder)
        path = os.path.join(folder, self.args.case_name + '_clusters.csv')
        text_df.to_csv(path, encoding='utf-8', index=False)
        path = os.path.join(folder, self.args.case_name + '_clusters.json')
        text_df.to_json(path, orient='records')
        # print(text_df)

    # Update iterative clustering scores with individual Silhouette scores
    def update_iterative_article_cluster_results(self):
        folder = os.path.join('output', self.args.case_name)
        # Load corpus
        # corpus_path = os.path.join(folder, 'iteration', self.args.case_name + '_clusters.json')
        # corpus = pd.read_json(corpus_path).to_dict("records")
        folder_names = ['cluster_0', 'cluster_1', 'cluster_2', 'cluster_3', 'iteration', 'cluster_-1']
        for folder_name in folder_names:
            iterative_folder = os.path.join(folder, folder_name)
            try:
                ArticleClusterTermTFIDFUtility.update_clustering_scores(iterative_folder, self.model)
                # Load the updated iterative clustering summary
                path = os.path.join(iterative_folder, 'cluster_terms', 'iterative_clusters',
                                    'AIMLUrbanStudyCorpus_iterative_summary.json')
                iterative_clusters = pd.read_json(path).to_dict("records")
                # Load the cluster results
                path = os.path.join(iterative_folder, self.args.case_name + '_cluster_terms_key_phrases_LDA_topics.json')
                cluster_results = pd.read_json(path).to_dict("records")
                for result in cluster_results:
                    found_cluster = next(c for c in iterative_clusters if np.array_equal(c['DocIds'], result['DocIds']))
                    assert found_cluster is not None, "Cannot find the article cluster"
                    result['Score'] = found_cluster['score']
                # Write output to article clusters
                df = pd.DataFrame(cluster_results)
                path = os.path.join(iterative_folder, self.args.case_name + '_cluster_terms_key_phrases_LDA_topics.csv')
                df.to_csv(path, encoding='utf-8', index=False)
                # Write article corpus to a json file
                path = os.path.join(iterative_folder, self.args.case_name + '_cluster_terms_key_phrases_LDA_topics.json')
                df.to_json(path, orient='records')
            except Exception as _err:
                print("Error occurred! {err}".format(err=_err))
                sys.exit(-1)

    # Collect all the article clusters < 40 articles as a single file
    def collect_article_cluster_results(self):
        folder = os.path.join('output', self.args.case_name)
        # Load corpus
        corpus_path = os.path.join(folder, 'iteration', self.args.case_name + '_clusters.json')
        corpus = pd.read_json(corpus_path).to_dict("records")
        folder_names = ['cluster_0', 'cluster_1', 'cluster_2', 'cluster_3', 'iteration', 'cluster_-1']
        cluster_results = list()
        article_results = list()
        current_cluster_no = 1
        for folder_name in folder_names:
            try:
                iterative_folder = os.path.join(folder, folder_name)
                cluster_path = os.path.join(iterative_folder,
                                            self.args.case_name + '_cluster_terms_key_phrases_LDA_topics.json')
                clusters = pd.read_json(cluster_path).to_dict("records")
                # filter cluster > 40 articles
                clusters = list(filter(lambda c: len(c['DocIds']) < 40, clusters))
                for cluster in clusters:
                    doc_ids = cluster['DocIds']
                    score = cluster['Score']
                    # Get all the articles
                    articles = list(filter(lambda a: a['DocId'] in doc_ids, corpus))
                    assert len(articles) < 40, "Article cluster > 40"
                    assert len(articles) > 0, "Article cluster is empty"
                    assert len(articles) == len(doc_ids), "Article cluster is not matched"
                    # Update the cluster and articles
                    for article in articles:
                        article['Cluster'] = current_cluster_no
                        article['Score'] = score
                    article_results = article_results + articles
                    cluster['Cluster'] = current_cluster_no
                    current_cluster_no = current_cluster_no + 1
                # Add the cluster results
                cluster_results = cluster_results + clusters
            except Exception as _err:
                print("Error occurred! {err}".format(err=_err))
                sys.exit(-1)
        # Sort article results by DocId
        article_results = sorted(article_results, key=lambda c: c['DocId'])
        # Write article corpus
        articles_df = pd.DataFrame(article_results, columns=['Cluster', 'Score', 'DocId', 'Cited by', 'Year',
                                                             'Document Type', 'Title', 'Abstract', 'Author Keywords',
                                                             'Authors', 'DOI', 'x', 'y'])
        articles_df = articles_df.rename(columns={"Cluster": "HDBSCAN_Cluster"})
        out_folder = os.path.join(folder, self.args.cluster_folder)
        Path(out_folder).mkdir(parents=True, exist_ok=True)
        # Write article corpus to csv file
        path = os.path.join(out_folder, self.args.case_name + '_clusters.csv')
        articles_df.to_csv(path, encoding='utf-8', index=False)
        # Write article corpus to a json file
        path = os.path.join(out_folder, self.args.case_name + '_clusters.json')
        articles_df.to_json(path, orient='records')

    # Derive the distinct from each cluster of documents
    def derive_cluster_terms_by_TF_IDF(self):
        try:
            term_folder = os.path.join('output', self.args.case_name, self.args.cluster_folder, 'cluster_terms')
            # Get the cluster docs
            path = os.path.join('output', self.args.case_name, self.args.cluster_folder,
                                self.args.case_name + '_clusters.json')
            # Load the documents clustered by
            clustered_doc_df = pd.read_json(path)
            # Update text column
            clustered_doc_df['Text'] = clustered_doc_df['Title'] + ". " + clustered_doc_df['Abstract']
            # Group the documents and doc_id by clusters
            docs_per_cluster_df = clustered_doc_df.groupby(['HDBSCAN_Cluster'], as_index=False) \
                .agg({'DocId': lambda doc_id: list(doc_id), 'Text': lambda text: list(text),
                      'Score': "mean"})
            # Get top 100 topics (1, 2, 3 grams) for each cluster
            n_gram_term_list = ArticleClusterTermTFIDFUtility.get_n_gram_terms('HDBSCAN_Cluster',
                                                                               docs_per_cluster_df,
                                                                               term_folder)
            results = []
            for i, cluster in docs_per_cluster_df.iterrows():
                try:
                    cluster_no = cluster['HDBSCAN_Cluster']
                    score = cluster['Score']
                    doc_ids = cluster['DocId']
                    doc_texts = cluster['Text']
                    result = {"Cluster": cluster_no, "Score": score, 'NumDocs': len(doc_ids), 'DocIds': doc_ids}
                    n_gram_terms = []
                    # Collect the topics of 1 gram, 2 gram and 3 gram
                    for n_gram_range in [1, 2]:
                        n_gram = next(n_gram for n_gram in n_gram_term_list
                                      if n_gram['n_gram'] == n_gram_range)
                        # Collect top 300 terms
                        cluster_terms = n_gram['terms'][str(cluster_no)][:300]
                        # Create a mapping between the topic and its associated articles (doc)
                        doc_per_term = ArticleClusterTermTFIDFUtility.group_docs_by_terms(n_gram_range,
                                                                                          doc_ids, doc_texts,
                                                                                          cluster_terms)
                        n_gram_type = 'Term-' + str(n_gram_range) + '-gram'
                        result[n_gram_type] = doc_per_term
                        n_gram_terms += doc_per_term
                    result['Term-N-gram'] = ArticleClusterTermTFIDFUtility.merge_n_gram_terms(n_gram_terms)
                    results.append(result)
                    print('Derive term of cluster #{no}'.format(no=cluster_no))
                except Exception as _err:
                    print("Error occurred! {err}".format(err=_err))
                    sys.exit(-1)
            # Write the result to csv and json file
            cluster_df = pd.DataFrame(results, columns=['Cluster', 'Score', 'NumDocs', 'DocIds',
                                                        'Term-1-gram', 'Term-2-gram', 'Term-N-gram'])
            folder = os.path.join(term_folder, 'TF_IDF_Terms')
            Path(folder).mkdir(parents=True, exist_ok=True)
            path = os.path.join(folder, 'TF-IDF_cluster_term.csv')
            cluster_df.to_csv(path, encoding='utf-8', index=False)
            # # # Write to a json file
            path = os.path.join(folder, 'TF-IDF_cluster_term.json')
            cluster_df.to_json(path, orient='records')
            print('Output terms per cluster to ' + path)
        except Exception as err:
            print("Error occurred! {err}".format(err=err))

    #  Summarize cluster terms and output to a single file
    def summarize_cluster_terms(self):
        def get_cluster_terms(terms, top_n=10):
            # Get top 10 terms
            cluster_terms = terms[:top_n]
            # Sort the cluster terms by number of docs and freq
            cluster_terms = sorted(cluster_terms, key=lambda t: (len(t['doc_ids']), t['freq']), reverse=True)
            # print(cluster_terms)
            return cluster_terms

        try:
            term_folder = os.path.join('output', self.args.case_name, self.args.cluster_folder, 'cluster_terms')
            # Load cluster topics
            path = os.path.join(term_folder, 'TF_IDF_Terms', 'TF-IDF_cluster_term.json')
            cluster_df = pd.read_json(path)
            # Write out to csv and json file
            cluster_df = cluster_df[['Cluster', 'Score', 'NumDocs', 'DocIds', 'Term-N-gram']]
            cluster_df.rename(columns={'Term-N-gram': 'Terms'}, inplace=True)
            cluster_df['Terms'] = cluster_df['Terms'].apply(lambda terms: get_cluster_terms(terms, 10))
            # # Output cluster df to csv or json file
            path = os.path.join(term_folder, self.args.case_name + '_TF-IDF_cluster_terms.csv')
            cluster_df.to_csv(path, encoding='utf-8', index=False)
            path = os.path.join(term_folder, self.args.case_name + '_TF-IDF_cluster_terms.json')
            cluster_df.to_json(path, orient='records')
            # Output a summary of top 10 Topics of each cluster
            clusters = cluster_df.to_dict("records")
            summary_df = cluster_df.copy(deep=True)
            total = summary_df['NumDocs'].sum()
            summary_df['Percent'] = list(map(lambda c: c['NumDocs'] / total, clusters))
            summary_df['Terms'] = list(
                map(lambda c: ", ".join(list(map(lambda t: t['term'], c['Terms'][:10]))), clusters))
            summary_df = summary_df.reindex(columns=['Cluster', 'Score', 'NumDocs', 'Percent', 'DocIds', 'Terms'])
            # Output the summary as csv
            path = os.path.join(term_folder, self.args.case_name + '_TF-IDF_cluster_terms_summary.csv')
            summary_df.to_csv(path, encoding='utf-8', index=False)
        except Exception as err:
            print("Error occurred! {err}".format(err=err))


# Main entry
if __name__ == '__main__':
    try:
        # cluster_no = 2
        # last_iteration = 2
        # ct = ClusterTermTFIDF(last_iteration, cluster_no)
        # ct.collect_iterative_cluster_results()
        # ct.output_iterative_cluster_results()
        ct = ArticleClusterTermTFIDF()
        # ct.update_iterative_article_cluster_results()
        ct.collect_article_cluster_results()
        ct.derive_cluster_terms_by_TF_IDF()
        ct.summarize_cluster_terms()
    except Exception as err:
        print("Error occurred! {err}".format(err=err))
