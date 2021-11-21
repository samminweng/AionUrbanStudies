import csv
import getpass
import os
import nltk
from nltk import word_tokenize, sent_tokenize
import pandas as pd
import numpy as np
# Load function words
from sentence_transformers import util
from sklearn.metrics.pairwise import cosine_similarity

# Set NLTK data path
nltk_path = os.path.join('/Scratch', getpass.getuser(), 'nltk_data')
if os.name == 'nt':
    nltk_path = os.path.join("C:", os.sep, "Users", getpass.getuser(), "nltk_data")
nltk.download('punkt', download_dir=nltk_path)
nltk.download('stopwords', download_dir=nltk_path)
# Append NTLK data path
nltk.data.path.append(nltk_path)


# Helper function for cluster Similarity
class ClusterSimilarityUtility:
    @staticmethod
    def compute_similarity_matrix_topics(cluster_topics1, cluster_topics2):
        c_no_1 = cluster_topics1['cluster_no']
        c_no_2 = cluster_topics2['cluster_no']
        # Compute the similarity by average vectors
        topic_vectors1 = cluster_topics1['topic_vectors']
        topic_vectors2 = cluster_topics2['topic_vectors']
        # Create a matrix
        similarity_matrix = np.zeros((len(topic_vectors1), len(topic_vectors2)))

        # Iterate the topics in cluster_1
        for i, vector1 in enumerate(topic_vectors1):
            for j, vector2 in enumerate(topic_vectors2):
                sim = cosine_similarity([vector1, vector2])[0][1]
                similarity_matrix[i, j] = sim
                # print("Similarity between {t1} and {t2} = {sim:.2f}".format(t1=cluster_topics1['topics'][i],
                #                                                             t2=cluster_topics2['topics'][j],
                #                                                             sim=sim))
        sim_df = pd.DataFrame(similarity_matrix, index=cluster_topics1['topics'], columns=cluster_topics2['topics'])
        sim_df = sim_df.round(2)  # Round each similarity to 2 decimal
        # Write to out
        _path = os.path.join('output', 'topic', 'matrix',
                             'UrbanStudyCorpus_HDBSCAN_similarity_matrix_' + str(c_no_1) + '_' + str(c_no_2) + '.csv')
        sim_df.to_csv(_path, encoding='utf-8')
        return similarity_matrix

    # Get the cluster topics
    @staticmethod
    def compute_cluster_topics(cluster, corpus_docs, clusters, model, top_n=10):
        try:
            # Collect all the top N topics (words and vectors)
            cluster_topic_vectors = []
            for cluster in clusters:
                cluster_no = cluster['Cluster']
                # Get top cluster topics
                top_n_cluster_topics = cluster['TopicN-gram'][:top_n]
                # Collect all topics and the topic vectors
                topic_vectors = model.encode(top_n_cluster_topics)
                topic_vectors = topic_vectors.numpy()
                # Add the
                cluster_topic_vectors.append({'cluster_no': cluster_no,
                                              'topics': top_n_cluster_topics,
                                              'topic_vectors': topic_vectors})
            # # Write out the cluster topic information
            df = pd.DataFrame(cluster_topic_vectors, columns=['cluster_no', 'topics', 'topic_vectors'])
            _path = os.path.join('output', 'topic', 'UrbanStudyCorpus_HDBSCAN_topic_vectors.csv')
            df.to_csv(_path, encoding='utf-8', index=False)
            return cluster_topic_vectors
        except Exception as err:
            print("Error occurred! {err}".format(err=err))

    # # Get the title vectors of all articles in a cluster
    @staticmethod
    def find_top_n_similar_title(cluster_no, corpus_docs, clusters, model, top_k=30):
        # Clean licensee sentences from a sentence
        def clean_sentence(text):
            sentences = sent_tokenize(text)
            # Preprocess the sentence
            cleaned_sentences = list()  # Skip copy right sentence
            for sentence in sentences:
                if u"\u00A9" not in sentence.lower() and 'licensee' not in sentence.lower() \
                        and 'copyright' not in sentence.lower() and 'rights reserved' not in sentence.lower():
                    try:
                        cleaned_words = word_tokenize(sentence.lower())
                        # Keep alphabetic characters only and remove the punctuation
                        cleaned_sentences.append(" ".join(cleaned_words))  # merge tokenized words into sentence
                    except Exception as err:
                        print("Error occurred! {err}".format(err=err))
            return ". ".join(cleaned_sentences)

        # Get the cluster no of a doc
        def get_cluster_no_doc(_doc_id, _clusters):
            _cluster = next(filter(lambda c: _doc_id in c['DocIds'], _clusters), None)
            if _cluster:
                return _cluster['Cluster']
            else:
                return

        # Process the source titles
        src_cluster = next(filter(lambda c: c['Cluster'] == cluster_no, clusters))
        # Get all the docs in src cluster (such as #15)
        src_docs = list(filter(lambda d: d['DocId'] in src_cluster['DocIds'], corpus_docs))
        results = []
        # Go through each doc in src cluster
        for i, src_doc in enumerate(src_docs):
            try:
                src_doc_id = src_doc['DocId']
                # Get the all the docs except for 'src' doc id
                target_docs = list(filter(lambda d: d['DocId'] != src_doc_id, corpus_docs))
                target_texts = list(map(lambda d: clean_sentence(d['Title'] + ". " + d['Abstract']),
                                        target_docs))
                # Perform semantic search (cosine similarity) to find top K (30) similar titles in corpus
                src_vector = model.encode(clean_sentence(src_doc['Title'] + ". " + src_doc['Abstract']),
                                          convert_to_tensor=True)
                target_vectors = model.encode(target_texts, convert_to_tensor=True)
                hits = util.semantic_search(src_vector, target_vectors, top_k=top_k)[0]
                # Collect top five similar titles for 'src_title'
                result = {"DocId": src_doc_id, "Title": src_doc['Title']}
                similar_papers = []
                for hit in hits:
                    t_id = hit['corpus_id']
                    target_doc = target_docs[t_id]
                    score = hit['score']
                    target_doc_id = target_doc["DocId"]
                    similar_papers.append({
                        'DocId': target_doc_id,
                        'Cluster': get_cluster_no_doc(target_doc_id, clusters),
                        'Title': target_doc['Title'],
                        'Score': round(score, 2)
                    })
                result['Similar_Papers'] = similar_papers
                results.append(result)
            except Exception as err:
                print("Error occurred! {err}".format(err=err))
        # # Write out the similarity title information
        df = pd.DataFrame(results)
        # # Write to out
        path = os.path.join('output', 'similarity', 'cluster',
                            'UrbanStudyCorpus_HDBSCAN_paper_similarity_' + str(cluster_no) + '.json')
        df.to_json(path, orient='records')
        return results

    # Write the results to a csv file
    @staticmethod
    def write_to_title_csv_file(cluster_no, top_k=30):
        # # Get the most occurring clusters
        def get_most_occurring_cluster(_similar_papers):
            _cluster_occ = list()
            for _similar_paper in _similar_papers:
                _c_no = _similar_paper['Cluster']
                _doc_id = _similar_paper['DocId']
                _c_occ = list(filter(lambda _occ: _occ['Cluster'] == _c_no, _cluster_occ))
                if len(_c_occ) == 0:
                    _doc_ids = list()
                    _doc_ids.append(_doc_id)
                    _cluster_occ.append({'Cluster': _c_no, "DocIds": _doc_ids})
                else:
                    _occ = _c_occ[0]
                    _occ['DocIds'].append(_doc_id)
            # Sort cluster occ by the number
            _sorted_cluster_occ = sorted(_cluster_occ, key=lambda _occ: len(_occ['DocIds']), reverse=True)
            return _sorted_cluster_occ

        # Load
        path = os.path.join('output', 'similarity', 'cluster',
                            'UrbanStudyCorpus_HDBSCAN_paper_similarity_' + str(cluster_no) + '.json')
        df = pd.read_json(path)
        results = df.to_dict("records")
        # Sort results by the most similar score from low to high
        sorted_results = sorted(results, key=lambda r: r['Similar_Papers'][0]['Score'])
        # Specify utf-8 as encoding
        # csv_file = open(out_path, "w", newline='', encoding='utf-8')
        # rows = list()
        # cluster_no_set = set()
        top_similar_papers = []
        # Header
        for item in sorted_results:
            top_st = item['Similar_Papers'][0]
            doc_id = item['DocId']
            title = item['Title']
            score = "{:.2f}".format(top_st['Score'])
            similar_doc_id = top_st['DocId']
            similar_cluster_no = top_st['Cluster']
            similar_title = top_st['Title']
            # cluster_no_set.add(similar_cluster_no)
            cluster_occ = get_most_occurring_cluster(item['Similar_Papers'])
            is_match = bool(similar_cluster_no == cluster_occ[0]['Cluster'])
            assert len(cluster_occ) >= 3
            result = {
                'doc_id': doc_id, 'title': title,
                'score': score, 'cluster': "#" + str(similar_cluster_no),
                'similar_doc_id': similar_doc_id,
                'similar_title': similar_title,
                'top1_occ_cluster': "#" + str(cluster_occ[0]['Cluster']), 'top1_count': len(cluster_occ[0]['DocIds']),
                'top2_occ_cluster': "#" + str(cluster_occ[1]['Cluster']), 'top2_count': len(cluster_occ[1]['DocIds']),
                'top3_occ_cluster': "#" + str(cluster_occ[2]['Cluster']), 'top3_count': len(cluster_occ[2]['DocIds']),
                'is_match': is_match}
            top_similar_papers.append(result)
            # Write the summary
        df = pd.DataFrame(top_similar_papers)
        out_path = os.path.join('output', 'similarity',
                                'UrbanStudyCorpus_HDBSCAN_similar_papers_' + str(cluster_no) + '_short.csv')
        df.to_csv(out_path, index=False, encoding='utf-8')
        # # Display the cluster no list
        # cluster_no_list = list(cluster_no_set)
        # cluster_no_list.sort()
        # print(cluster_no_list)

            # rows.append([doc_id, title, score, similar_cluster_no,
            #              similar_doc_id, similar_title])
            # for _i in range(1, top_k):
            #     st = item['Similar_Titles'][_i]
            #     rows.append(["", "", "{:.2f}".format(st['Score']), st['Cluster'], st['DocId'], st['Title']])
            # Add blank row
            # rows.append([])
            # # Write row to csv_file
            # writer = csv.writer(csv_file)
            # for row in rows:
            #     writer.writerow(row)
            # csv_file.close()
            # Give a summary
