import csv
import getpass
import itertools
import os
import re
import string
from functools import reduce
from pathlib import Path

import hdbscan
import nltk
import umap
from nltk import word_tokenize, sent_tokenize, ngrams, pos_tag
import pandas as pd
import numpy as np
# Load function words
from nltk.corpus import stopwords
from sentence_transformers import util
from sklearn.metrics.pairwise import cosine_similarity, pairwise_distances
from nltk.stem import WordNetLemmatizer

# Set NLTK data path
from BERTModelDocClusterUtility import BERTModelDocClusterUtility

nltk_path = os.path.join('/Scratch', getpass.getuser(), 'nltk_data')
if os.name == 'nt':
    nltk_path = os.path.join("C:", os.sep, "Users", getpass.getuser(), "nltk_data")
nltk.download('punkt', download_dir=nltk_path)
nltk.download('wordnet', download_dir=nltk_path)
nltk.download('stopwords', download_dir=nltk_path)
nltk.download('averaged_perceptron_tagger', download_dir=nltk_path)  # POS tags
# Append NTLK data path
nltk.data.path.append(nltk_path)


# Load the lemma.n file to store the mapping of singular to plural nouns
def load_lemma_nouns():
    _lemma_nouns = {}
    path = os.path.join('data', 'lemma.n')
    f = open(path, 'r')
    lines = f.readlines()
    for line in lines:
        words = line.rstrip().split("->")  # Remove trailing new line char and split by '->'
        plural_word = words[1]
        if '.,' in plural_word:  # Handle multiple plural forms and get the last one as default plural form
            plural_word = plural_word.split('.,')[-1]
        singular_word = words[0]
        _lemma_nouns[plural_word] = singular_word
    return _lemma_nouns


stop_words = list(stopwords.words('english'))
lemma_nouns = load_lemma_nouns()


# Helper function for cluster Similarity
class KeyPhraseUtility:
    # Clean licence statement texts
    @staticmethod
    def clean_sentence(text):
        # Split the words 'within/near'
        def split_words(_words):
            _out_words = list()
            for _word in _words:
                if matches := re.match(r'(\w+)/(\w+)', _word):
                    _out_words.append(matches.group(1))
                    _out_words.append(matches.group(2))
                elif re.match(r"('\w+)|(\w+')", _word):
                    _out_words.append(_word.replace("'", ""))
                else:
                    _out_words.append(_word)
            return _out_words

        # Change plural nouns to singular nouns using lemmatizer
        def convert_singular_words(_words, _lemmatiser):
            # Tag the words with part-of-speech tags
            _pos_tags = nltk.pos_tag(_words)
            # Convert plural word to singular
            singular_words = []
            for i, (_word, _pos_tag) in enumerate(_pos_tags):
                try:
                    # Lowercase 1st char of the firs word
                    # if i == 0:
                    #    _word = _word[0].lower() + _word[1:len(_word)]
                    # NNS indicates plural nouns and convert the plural noun to singular noun
                    if _pos_tag == 'NNS':
                        singular_word = _lemmatiser.lemmatize(_word.lower())
                        if _word[0].isupper():  # Restore the uppercase
                            singular_word = singular_word.capitalize()  # Upper case the first character
                        singular_words.append(singular_word)
                    else:
                        # Check if the word in lemma list
                        if _word.lower() in lemma_nouns:
                            try:
                                singular_word = lemma_nouns[_word.lower()]
                                if _word[0].isupper():  # Restore the uppercase
                                    singular_word = singular_word.capitalize()  # Upper case the first character
                                singular_words.append(singular_word)
                            except Exception as _err:
                                print("Error occurred! {err}".format(err=_err))
                        else:
                            singular_words.append(_word)
                except Exception as _err:
                    print("Error occurred! {err}".format(err=_err))
            # Return all lemmatized words
            return singular_words

        lemmatizer = WordNetLemmatizer()
        sentences = sent_tokenize(text)
        # Preprocess the sentence
        cleaned_sentences = list()  # Skip copy right sentence
        for sentence in sentences:
            if u"\u00A9" not in sentence.lower() and 'licensee' not in sentence.lower() \
                    and 'copyright' not in sentence.lower() and 'rights reserved' not in sentence.lower():
                try:
                    words = split_words(word_tokenize(sentence))
                    if len(words) > 0:
                        # Convert the plural words into singular words
                        cleaned_words = convert_singular_words(words, lemmatizer)
                        cleaned_sentences.append(cleaned_words)  # merge tokenized words into sentence
                except Exception as err:
                    print("Error occurred! {err}".format(err=err))
        return cleaned_sentences  # Convert a list of clean sentences to the text

    @staticmethod
    # Generate n-gram of a text and avoid stop
    def generate_n_gram_candidates(sentences, n_gram_range):
        def _is_qualified(_n_gram):  # _n_gram is a list of tuple (word, tuple)
            try:
                qualified_tags = ['NN', 'NNS', 'JJ', 'NNP']
                # # Check if there is any noun
                nouns = list(filter(lambda _n: _n[1].startswith('NN'), _n_gram))
                if len(nouns) == 0:
                    return False
                # Check the last word is a nn or nns
                if _n_gram[-1][1] not in ['NN', 'NNS']:
                    return False
                # Check if all words are not stop word or punctuation or non-words
                for _i, _n in enumerate(_n_gram):
                    _word = _n[0]
                    _pos_tag = _n[1]
                    if bool(re.search(r'\d|[^\w]', _word.lower())) or _word.lower() in string.punctuation or \
                            _word.lower() in stop_words or _pos_tag not in qualified_tags:
                        return False
                # n-gram is qualified
                return True
            except Exception as err:
                print("Error occurred! {err}".format(err=err))

        candidates = list()
        # Extract n_gram from each sentence
        for i, sentence in enumerate(sentences):
            try:
                pos_tags = pos_tag(sentence)
                # Pass pos tag tuple (word, pos-tag) of each word in the sentence to produce n-grams
                n_grams = list(ngrams(pos_tags, n_gram_range))
                # Filter out not qualified n_grams that contain stopwords or the word is not alpha_numeric
                for n_gram in n_grams:
                    if _is_qualified(n_gram):
                        n_gram_text = " ".join(list(map(lambda n: n[0], n_gram)))
                        # Check if candidates exist in the list
                        found = next((c for c in candidates if c.lower() == n_gram_text.lower()), None)
                        if not found:
                            candidates.append(n_gram_text)  # Convert n_gram (a list of words) to a string
            except Exception as err:
                print("Error occurred! {err}".format(err=err))
        return candidates

    # Find top K key phrase similar to the paper
    # Ref: https://www.sbert.net/examples/applications/semantic-search/README.html
    @staticmethod
    def collect_top_key_phrases(model, doc_text, candidates, top_k=5):
        try:
            if len(candidates) == 0:
                return []

            # Encode cluster doc and keyword candidates into vectors for comparing the similarity
            candidate_vectors = model.encode(candidates, convert_to_numpy=True)
            doc_vector = model.encode([doc_text], convert_to_numpy=True)  # Convert the numpy array
            # Compute the distance of doc vector and each candidate vector
            distances = cosine_similarity(doc_vector, candidate_vectors)
            # Select top key phrases based on the distance score
            top_key_phrases = list()
            min_length = min(len(candidates), top_k)  # Get the minimal
            # Get top 5 candidate of smallest distances or all the candidates if 4 or few
            top_distances = distances.argsort()[0][-min_length:]
            for c_index in top_distances:
                candidate = candidates[c_index]
                distance = distances[0][c_index]
                # vector = candidate_vectors[c_index]
                found = next((kp for kp in top_key_phrases if kp['key-phrase'].lower() == candidate.lower()), None)
                if not found:
                    top_key_phrases.append({'key-phrase': candidate, 'score': distance})
            # Sort the phrases by scores
            top_key_phrases = sorted(top_key_phrases, key=lambda k: k['score'], reverse=True)
            return top_key_phrases
        except Exception as err:
            print("Error occurred! {err}".format(err=err))

    @staticmethod
    # Write the key phrases of each cluster to a csv file
    def output_key_phrases_by_cluster(key_phrase_list, cluster_no, folder):
        try:
            df = pd.DataFrame(key_phrase_list, columns=['DocId'])
            df['No'] = range(1, len(df) + 1)
            # Map the list of key phrases (dict) to a list of strings
            # Map the nested dict to a list of key phrases (string only)
            df['key-phrases'] = list(map(lambda k: [kp['key-phrase'] for kp in k['key-phrases']], key_phrase_list))
            df = df[['No', 'DocId', 'key-phrases']]  # Re-order the columns

            # Path(folder).mkdir(parents=True, exist_ok=True)
            path = os.path.join(folder, 'top_key_phrases_cluster_#' + str(cluster_no) + '.csv')
            df.to_csv(path, encoding='utf-8', index=False)
            path = os.path.join(folder, 'top_key_phrases_cluster_#' + str(cluster_no) + '.json')
            df.to_json(path, orient='records')
            print("Output the key phrases of cluster #" + str(cluster_no))
        except Exception as _err:
            print("Error occurred! {err}".format(err=_err))

    # Get a list of unique key phrases from all papers
    @staticmethod
    def get_unique_doc_key_phrases(doc_key_phrases, all_key_phrases, top_k=5):
        try:
            if len(doc_key_phrases) < top_k:
                return doc_key_phrases

            unique_key_phrases = list()
            for key_phrase in doc_key_phrases:
                # find if key phrase exist in all key phrase list
                found = next((kp for kp in all_key_phrases
                              if kp['key-phrase'].lower() == key_phrase['key-phrase'].lower()), None)
                if not found:
                    unique_key_phrases.append(key_phrase)
                else:
                    print("Duplicated: " + found['key-phrase'])

            # Get top 5 key phrase
            unique_key_phrases = unique_key_phrases[:top_k]
            # assert len(_unique_key_phrases) == _top_k
            return unique_key_phrases
        except Exception as _err:
            print("Error occurred! {err}".format(err=_err))

    @staticmethod
    def group_key_phrases_with_best_result(cluster_no, parameter, doc_key_phrases, folder):
        # Collect the key phrases linked to the docs
        def get_doc_ids_by_group_key_phrases(_doc_key_phrases, _grouped_key_phrases):
            _doc_ids = list()
            for doc in _doc_key_phrases:
                # find if any doc key phrases appear in the grouped key phrases
                for key_phrase in doc['key-phrases']:
                    found = next((gkp for gkp in _grouped_key_phrases if gkp.lower() == key_phrase.lower()), None)
                    if found:
                        _doc_ids.append(doc['DocId'])
                        break
            return _doc_ids

        try:
            # Aggregate all the key phrases of each doc in a cluster as a single list
            key_phrases = reduce(lambda pre, cur: pre + cur['key-phrases'], doc_key_phrases, list())
            # Get the grouping labels of key phrases
            group_labels = parameter['group_labels']
            # Load key phrase and group labels
            df = pd.DataFrame()
            df['key-phrases'] = key_phrases
            df['group'] = group_labels
            # Output the summary of the grouped key phrase results
            group_df = df.groupby(by=['group'], as_index=False).agg({'key-phrases': lambda k: list(k)})
            # Output the summary results to a csv file
            group_df['cluster'] = cluster_no
            group_df['count'] = group_df['key-phrases'].apply(len)
            # Collect doc ids that contained the grouped key phrases
            group_key_phrases = group_df['key-phrases'].tolist()
            group_doc_ids = list(
                map(lambda group: get_doc_ids_by_group_key_phrases(doc_key_phrases, group), group_key_phrases))
            group_df['DocIds'] = group_doc_ids
            group_df['NumDocs'] = group_df['DocIds'].apply(len)
            group_df = group_df[
                ['cluster', 'group', 'count', 'key-phrases', 'NumDocs', 'DocIds']]  # Re-order the column list
            path = os.path.join(folder, 'group_key_phrases',
                                'top_key_phrases_cluster_#' + str(cluster_no) + '_best_grouping.csv')
            group_df.to_csv(path, encoding='utf-8', index=False)
            # Output the summary of best grouped key phrases to a json file
            path = os.path.join(folder, 'group_key_phrases',
                                'top_key_phrases_cluster_#' + str(cluster_no) + '_best_grouping.json')
            group_df.to_json(path, orient='records')
            print('Output the summary of grouped key phrase to ' + path)
            return group_df.to_dict("records")
        except Exception as err:
            print("Error occurred! {err}".format(err=err))

    # Cluster key phrases (vectors) using HDBSCAN clustering
    @staticmethod
    def group_key_phrase_experiments_by_HDBSCAN(key_phrases, cluster_no, model, folder):
        def collect_group_results(_results, _group_label):
            try:
                _found = next((r for r in _results if r['group'] == _group_label), None)
                if not _found:
                    _found = {'group': _group_label, 'count': 1}
                    _results.append(_found)
                else:
                    _found['count'] += 1
                # Sort the results
                _results = sorted(_results, key=lambda c: (c['count'], c['group']))
                return _results
            except Exception as _err:
                print("Error occurred! {err}".format(err=_err))

        try:
            # Convert the key phrases to vectors
            key_phrase_vectors = model.encode(key_phrases)
            results = list()
            for dimension in [10, 15, 30, 50, 80, 100, 150, 768]:
                for min_samples in [None] + list(range(1, 16)):
                    for min_cluster_size in [5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]:
                        for epsilon in [0.0]:
                            try:
                                parameter = {'min_cluster_size': min_cluster_size, 'min_samples': min_samples,
                                             'epsilon': epsilon}
                                if dimension == key_phrase_vectors.shape[1]:
                                    reduced_vectors = key_phrase_vectors
                                else:
                                    vector_list = key_phrase_vectors.tolist()
                                    # Reduce the doc vectors to specific dimension
                                    reduced_vectors = umap.UMAP(
                                        min_dist=0.0,
                                        n_components=dimension,
                                        random_state=42,
                                        metric="cosine").fit_transform(vector_list)
                                # Get vector dimension
                                dimension = reduced_vectors.shape[1]
                                # Compute the cosine distance/similarity for each doc vectors
                                distances = pairwise_distances(reduced_vectors, metric='cosine')
                                # # Pass key phrase vector to HDBSCAN for grouping
                                group_labels = hdbscan.HDBSCAN(min_cluster_size=parameter['min_cluster_size'],
                                                               min_samples=parameter['min_samples'],
                                                               cluster_selection_epsilon=parameter['epsilon'],
                                                               metric='precomputed').fit_predict(
                                    distances.astype('float64')).tolist()
                                group_results = reduce(lambda pre, cur: collect_group_results(pre, cur), group_labels,
                                                       list())
                                outlier_number = next((g['count'] for g in group_results if g['group'] == -1), 0)
                                if len(group_results) > 1:
                                    df = pd.DataFrame()
                                    df['groups'] = group_labels
                                    df['vectors'] = reduced_vectors.tolist()
                                    # Remove the outliers
                                    no_outlier_df = df[df['groups'] != -1]
                                    no_outlier_labels = no_outlier_df['groups'].tolist()
                                    no_outlier_vectors = np.vstack(no_outlier_df['vectors'].tolist())
                                    score = BERTModelDocClusterUtility.compute_Silhouette_score(no_outlier_labels,
                                                                                                no_outlier_vectors)
                                else:  # All key phrases are identified as outliers
                                    score = 'None'
                                # Output the result
                                result = {'cluster': "#" + str(cluster_no),
                                          'dimension': dimension,
                                          'min_samples': str(parameter['min_samples']),
                                          'min_cluster_size': parameter['min_cluster_size'],
                                          'epsilon': parameter['epsilon'],
                                          'total_groups': len(group_results),
                                          'outliers': outlier_number,
                                          'score': score,
                                          'group_result': group_results,
                                          'group_labels': group_labels}
                                results.append(result)
                                print(result)
                            except Exception as err:
                                print("Error occurred! {err}".format(err=err))
            # output the experiment results
            df = pd.DataFrame(results)
            path = os.path.join(folder, 'top_key_phrases_cluster_#' + str(cluster_no) + '_grouping_experiments.csv')
            df.to_csv(path, encoding='utf-8', index=False)
            path = os.path.join(folder, 'top_key_phrases_cluster_#' + str(cluster_no) + '_grouping_experiments.json')
            df.to_json(path, orient='records')
        except Exception as err:
            print("Error occurred! {err}".format(err=err))

    # # Get the title vectors of all articles in a cluster
    @staticmethod
    def find_top_n_similar_papers(cluster_no, corpus_docs, clusters, model, top_k=30):
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
                    except Exception as _err:
                        print("Error occurred! {err}".format(err=_err))
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
        path = os.path.join('output', 'similarity', 'similar_papers',
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

        # Write long version of paper similar
        def write_paper_similarity_long_version(_sorted_results):
            _path = os.path.join('output', 'key_phrases', 'similar_papers',
                                 'UrbanStudyCorpus_HDBSCAN_similar_papers_' + str(cluster_no) + '_long.csv')
            # Specify utf-8 as encoding
            csv_file = open(_path, "w", newline='', encoding='utf-8')
            rows = list()
            # Header
            rows.append(['doc_id', 'title', 'score', 'similar_cluster_no', 'similar_doc_id', 'similar_title'])
            for _doc in _sorted_results:
                most_similar_paper = _doc['Similar_Papers'][0]
                rows.append([_doc['DocId'], _doc['Title'], "{:.4f}".format(most_similar_paper['Score']),
                             most_similar_paper['Cluster'], most_similar_paper['DocId'], most_similar_paper['Title']])
                # Display the remaining 30
                for _i in range(1, top_k):
                    st = _doc['Similar_Papers'][_i]
                    rows.append(["", "", "{:.2f}".format(st['Score']), st['Cluster'], st['DocId'], st['Title']])
            # Add blank row
            rows.append([])
            # Write row to csv_file
            writer = csv.writer(csv_file)
            for row in rows:
                writer.writerow(row)
            csv_file.close()

        # Load
        path = os.path.join('output', 'key_phrases', 'similar_papers',
                            'UrbanStudyCorpus_HDBSCAN_paper_similarity_' + str(cluster_no) + '.json')
        df = pd.read_json(path)
        results = df.to_dict("records")
        # Sort results by the most similar score from low to high
        sorted_results = sorted(results, key=lambda r: r['Similar_Papers'][0]['Score'], reverse=True)
        # Write a long version of paper similarity
        write_paper_similarity_long_version(sorted_results)
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
        out_path = os.path.join('output', 'key_phrases', 'similar_papers',
                                'UrbanStudyCorpus_HDBSCAN_similar_papers_' + str(cluster_no) + '_short.csv')
        df.to_csv(out_path, index=False, encoding='utf-8')
        # # Display the cluster no list

    @staticmethod
    def most_similar_candidate(cluster_no, cluster_vector, c_vectors, candidates, top_n=5):
        keywords = []
        for index, (candidate, c_vector) in enumerate(zip(candidates, c_vectors)):
            score = cosine_similarity([cluster_vector], [c_vector])[0, 0]  # Get cosine similarity
            keywords.append({'keyword': candidate, 'score': score})
        # Sort the keywords by score
        keywords = sorted(keywords, key=lambda k: k['score'], reverse=True)
        top_keywords = keywords[:top_n]
        # Write the top_keywords as a csv file
        df = pd.DataFrame(top_keywords, columns=['score', 'keyword'])
        path = os.path.join('output', 'key_phrases', 'keywords', 'msc_top_keywords_cluster_' + str(cluster_no) + '.csv')
        df.to_csv(path, encoding='utf-8', index=False)
        return top_keywords

    @staticmethod
    def maximal_marginal_relevance(cluster_vector, c_vectors, candidates,
                                   top_n=30, diversity=0.5):
        try:
            # Extract similarity within words, and between words and the document
            word_doc_similarity = cosine_similarity(c_vectors, [cluster_vector])
            word_similarity = cosine_similarity(c_vectors, c_vectors)

            # Initialize candidates and already choose best keyword/key phrase
            keywords_idx = [np.argmax(word_doc_similarity)]
            candidates_idx = [i for i in range(len(candidates)) if i != keywords_idx[0]]
            # Re-rank the candidate key words
            for _ in range(top_n - 1):
                # Extract similarities within candidates and
                # between candidates and selected keywords/phrases
                candidate_similarities = word_doc_similarity[candidates_idx, :]
                target_similarities = np.max(word_similarity[candidates_idx][:, keywords_idx], axis=1)

                # Calculate MMR
                mmr = (1 - diversity) * candidate_similarities - diversity * target_similarities.reshape(-1, 1)
                mmr_idx = candidates_idx[np.argmax(mmr)]

                # Update keywords & candidates
                keywords_idx.append(mmr_idx)
                candidates_idx.remove(mmr_idx)

            # Collect all the top keywords
            top_keywords = list()
            for idx in keywords_idx:
                keyword = candidates[idx]
                score = word_doc_similarity[idx][0]
                top_keywords.append({'score': score, 'keyword': keyword})
            return top_keywords
        except Exception as _err:
            print("Error occurred! {err}".format(err=_err))
