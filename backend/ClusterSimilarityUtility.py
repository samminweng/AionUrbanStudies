import csv
import getpass
import itertools
import os
import re
import string

import nltk
from nltk import word_tokenize, sent_tokenize, ngrams, pos_tag
import pandas as pd
import numpy as np
# Load function words
from nltk.corpus import stopwords
from sentence_transformers import util
from sklearn.metrics.pairwise import cosine_similarity
from nltk.stem import WordNetLemmatizer

# Set NLTK data path
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
class ClusterSimilarityUtility:
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
                    if i == 0:
                        _word = _word[0].lower() + _word[1:len(_word)]
                    # NNS indicates plural nouns and convert the plural noun to singular noun
                    if _pos_tag == 'NNS':
                        singular_word = _lemmatiser.lemmatize(_word.lower())
                        if _word[0].isupper():  # Restore the uppercase
                            singular_word = singular_word.capitalize()      # Upper case the first character
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
        def _is_qualified(_n_gram):     # _n_gram is a list of tuple (word, tuple)
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
        return candidates

    # Find top K key phrase similar to the paper
    # Ref: https://www.sbert.net/examples/applications/semantic-search/README.html
    @staticmethod
    def get_top_key_phrases(model, doc_text, candidates, top_k=5):
        try:
            # Encode cluster doc and keyword candidates into vectors for comparing the similarity
            candidate_vectors = model.encode(candidates, convert_to_numpy=True)
            doc_vector = model.encode([doc_text], convert_to_numpy=True)  # Convert the numpy array
            # Compute the distance of doc vector and each candidate vector
            distances = cosine_similarity(doc_vector, candidate_vectors)
            # Select top key phrases based on the distance score
            top_key_phrases = list()
            for index in distances.argsort()[0][-top_k:]:
                candidate = candidates[index]
                distance = distances[0][index]
                vector = candidate_vectors[index]
                top_key_phrases.append({'key-phrase': candidate, 'score': distance})
            # Sort the phrases by scores
            top_key_phrases = sorted(top_key_phrases, key=lambda k: k['score'], reverse=True)
            # Output to key phrase (word-only)
            return list(map(lambda k: k['key-phrase'], top_key_phrases))
        except Exception as err:
            print("Error occurred! {err}".format(err=err))

    # Find the duplicate papers in the corpus
    @staticmethod
    def scan_duplicate_articles():
        try:
            corpus_df = pd.read_csv(os.path.join('data', 'UrbanStudyCorpus.csv'))
            corpus = corpus_df.to_dict("records")
            duplicate_doc_ids = set()
            for article in corpus:
                doc_id = article['DocId']
                title = article['Title']
                # Find if other article has the same title and author names
                same_articles = list(filter(lambda a: a['Title'].lower().strip() == title.lower().strip() and
                                                      a['DocId'] > doc_id, corpus))
                if len(same_articles):
                    for sa in same_articles:
                        duplicate_doc_ids.add(sa['DocId'])
            return list(duplicate_doc_ids)
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
            _path = os.path.join('output', 'similarity', 'similar_papers',
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
        path = os.path.join('output', 'similarity', 'similar_papers',
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
        out_path = os.path.join('output', 'similarity', 'similar_papers',
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
        path = os.path.join('output', 'similarity', 'keywords', 'msc_top_keywords_cluster_' + str(cluster_no) + '.csv')
        df.to_csv(path, encoding='utf-8', index=False)
        return top_keywords

    # Find the best combination that produce the lower similarity among keywords
    @staticmethod
    def maximize_sum_similarity(cluster_vector, c_vectors, candidates, top_n=5, num_candidates=20):
        try:
            # Compute the similarity between cluster vector and each candidate vector
            keywords = []
            for index, (candidate, c_vector) in enumerate(zip(candidates, c_vectors)):
                score = cosine_similarity([cluster_vector], [c_vector])[0, 0]  # Get cosine similarity
                keywords.append({'keyword': candidate, 'score': score, 'index': index,
                                 'vector': c_vector})
            # Sort the keywords by score
            keywords = sorted(keywords, key=lambda k: k['score'], reverse=True)
            keyword_candidates = keywords[:num_candidates]
            # Compute the similarity between candidate vectors
            kc_vectors = list(map(lambda k: k['vector'], keyword_candidates))
            # Get top number of candidate as potential keywords based on cosine similarity of cluster vector
            # Similarity between top keywords words
            kc_similarity = cosine_similarity(kc_vectors, kc_vectors)
            # Calculate the combination of words that are the least similar to each other
            min_sim = np.inf  # Start positive infinity
            best_comb = None
            for combination in itertools.combinations(range(len(keyword_candidates)), top_n):
                # Sum up the similarity of top keywords
                total_sim = sum([kc_similarity[i][j] for i in combination for j in combination if i != j])
                if total_sim < min_sim:  # if the combination has the lower similarity
                    best_comb = combination
                    min_sim = total_sim
            # Return a list of top and reverse the list
            top_keywords = [keyword_candidates[index] for index in best_comb]
            top_keywords.reverse()
            # Write the top_keywords as a csv file
            df = pd.DataFrame(top_keywords, columns=['score', 'keyword'])
            path = os.path.join('output', 'similarity', 'keywords',
                                'mss_top_keywords_cluster_' + str(num_candidates) + '.csv')
            df.to_csv(path, encoding='utf-8')
            return top_keywords
        except Exception as _err:
            print("Error occurred! {err}".format(err=_err))

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
