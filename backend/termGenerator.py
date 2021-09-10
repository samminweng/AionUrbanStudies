import json
import os.path
from argparse import Namespace
import pandas as pd
from nltk import BigramCollocationFinder, word_tokenize
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from Utility import Utility
import nltk

path = os.path.join('/Scratch', 'mweng', 'nltk_data')  # Specify the path of NLTK data
# Download all the necessary NLTK data
nltk.download('stopwords', download_dir=path)
nltk.download('punkt', download_dir=path)
nltk.download('wordnet', download_dir=path)
nltk.download('averaged_perceptron_tagger', download_dir=path)
nltk.download('words', download_dir=path)
nltk.data.path.append(path)


class TermGenerator:
    def __init__(self):
        self.args = Namespace(
            case_name='UrbanStudyCorpus',
            path='data'
        )
        self.stopwords = list(stopwords.words('english'))
        self.stopwords.append("also")  # add extra stop words
        # Load the function word
        path = os.path.join('data', 'Function_Words.csv')
        df = pd.read_csv(path)
        self.function_words = df['Function Word'].tolist()
        # print(self.function_words)

    # Extract important words from scikit TFIDF
    def collect_terms_from_TFIDF(self):
        path = os.path.join('data', self.args.case_name + '.csv')
        text_df = pd.read_csv(path)
        # Write corpus to json file
        path = os.path.join('output', self.args.case_name + '.json')
        text_df.to_json(path, orient='records')
        print('Output the corpus to ' + path)
        records = list()
        corpus = list()
        # Search all the subject words
        for i, text in text_df.iterrows():
            try:
                document = Utility.create_document(text)
                corpus.append(document)
                record = {'DocId': text['DocId'], 'Year': text['Year'], 'Title': text['Title'],
                          'Abstract': text['Abstract'], 'AuthorKeywords': text['Author Keywords']}
                records.append(record)
            except Exception as err:
                print("Error occurred! {err}".format(err=err))
        for n_gram_type in ['bi-gram']:
            key_term_lists = Utility.extract_terms_from_TFIDF(n_gram_type, corpus, self.stopwords, self.function_words)
            for index, key_terms in enumerate(key_term_lists):
                records[index]['KeyTerms'] = key_terms

        # Write the output to a file
        df = pd.DataFrame(records, columns=["DocId", "Year", "KeyTerms", "Title", "Abstract", 'AuthorKeywords'])
        path = os.path.join('output', self.args.case_name + '_doc_terms.csv')
        df.to_csv(path, encoding='utf-8', index=False)
        # # Write to a json file
        path = os.path.join('output', self.args.case_name + '_doc_terms.json')
        df.to_json(path, orient='records')
        print('Output keywords/phrases to ' + path)

    # Collect term frequency
    def collect_term_frequency(self):
        path = os.path.join('output', self.args.case_name + '_key_terms.json')
        text_df = pd.read_json(path)
        term_dict = {'uni_gram': {},
                     'bi_gram': {},
                     'tri_gram': {}
                     }
        for i, text in text_df.iterrows():
            try:
                term_dict['uni_gram'] = Utility.collect_term_freq_docID('uni_gram', term_dict['uni_gram'],
                                                                        text['Key 1-Words'],
                                                                        text)
                term_dict['bi_gram'] = Utility.collect_term_freq_docID('bi_gram', term_dict['bi_gram'],
                                                                       text['Key 2-Words'],
                                                                       text)
                term_dict['tri_gram'] = Utility.collect_term_freq_docID('tri_gram', term_dict['tri_gram'],
                                                                        text['Key 3-Words'],
                                                                        text)
            except Exception as err:
                print("Error occurred! {err}".format(err=err))
        # Sort the terms by total number of docIDs
        sort_freq_dict = {}
        max_length = 0
        for n_gram_type in ['uni_gram', 'bi_gram', 'tri_gram']:
            # Sort the terms by number of documents
            sort_freq_dict[n_gram_type] = sorted(term_dict[n_gram_type].items(),
                                                 key=lambda item: len(item[1]['docIDs']),
                                                 reverse=True)

            if max_length < len(sort_freq_dict[n_gram_type]):
                max_length = len(sort_freq_dict[n_gram_type])
        # Display all the term and frequency
        records = list()
        for i in range(max_length):
            record = {}
            for n_gram_type in ['uni_gram', 'bi_gram', 'tri_gram']:
                if i < len(sort_freq_dict[n_gram_type]):
                    record[n_gram_type] = sort_freq_dict[n_gram_type][i][0]
                    record[n_gram_type + "_freq"] = sort_freq_dict[n_gram_type][i][1]['freq']
                    record[n_gram_type + "_docIDs"] = len(sort_freq_dict[n_gram_type][i][1]['docIDs'])
                else:
                    record[n_gram_type] = ""
                    record[n_gram_type + "_freq"] = ""
                    record[n_gram_type + "_docIDs"] = ""
            records.append(record)
        df = pd.DataFrame(records, columns=['uni_gram', 'uni_gram_freq', 'uni_gram_docIDs',
                                            'bi_gram', 'bi_gram_freq', 'bi_gram_docIDs',
                                            'tri_gram', 'tri_gram_freq', 'tri_gram_docIDs'])
        path = os.path.join('output', self.args.case_name + '_term_freq.csv')
        df.to_csv(path, encoding='utf-8', index=False)
        # # Write to a json file
        path = os.path.join('output', self.args.case_name + '_term_freq.json')
        df.to_json(path, orient='records')
        print('Output keywords/phrases to ' + path)

    # Collect bigrams by Pointwise
    def collect_and_rank_collocations(self):
        path = os.path.join('data', self.args.case_name + '.csv')
        text_df = pd.read_csv(path)
        # Create NLTK bigram object
        bigram_measures = nltk.collocations.BigramAssocMeasures()
        # Search all the subject words
        documents = list()
        for i, text in text_df.iterrows():
            document = Utility.create_document(text)
            # Find all the bigrams from the document
            tokens = word_tokenize(document)
            documents.append(tokens)
        # Score and rank the collocations
        finder = BigramCollocationFinder.from_documents(documents)
        collocations = Utility.get_collocations(bigram_measures, finder, self.stopwords, self.function_words)
        records = list()
        for collocation in collocations:
            record = {'Collocation': collocation['collocation'], 'Score': collocation['score']}
            records.append(record)
        # Write the output to a file
        df = pd.DataFrame(records, columns=['Collocation', 'Score'])
        path = os.path.join('output', self.args.case_name + '_collocations_likelihood.csv')
        df.to_csv(path, encoding='utf-8', index=False)
        # # Write to a json file
        path = os.path.join('output', self.args.case_name + '_collocations_likelihood.json')
        df.to_json(path, orient='records')

    # Collect the relation between collocations and document IDs
    def collect_collocation_doc_ids(self):
        # Read the corpus
        text_df = pd.read_csv(os.path.join('data', self.args.case_name + '.csv'))
        # Read collocation
        col_df = pd.read_json(os.path.join('output', self.args.case_name + '_collocations_likelihood.json'))
        # Read doc term (TF-IDF)
        doc_term_df = pd.read_json(os.path.join('output', self.args.case_name + '_doc_terms.json'))
        # Collect all the relevant information
        collocations = col_df['Collocation'].tolist()
        scores = col_df['Score'].tolist()
        col_doc_dict = Utility.create_collocation_document(collocations, text_df)
        records = list()
        max_length = 15     # Max number of term map
        min_occurrence = 3  # Mininal number of paper that a collocation appears in the corpus
        for index, collocation in enumerate(collocations):
            score = scores[index]
            doc_ids = col_doc_dict[collocation]
            num_docs = len(doc_ids)
            if num_docs > min_occurrence:
                doc_ids_year = Utility.group_doc_ids_by_year(text_df, doc_ids)
                term_map = Utility.compute_term_map(doc_term_df, doc_ids)[:max_length]  # Store top frequent terms
                occurrences = Utility.compute_co_occurrence_terms(term_map)
                record = {'Collocation': collocation, 'Score': score,
                          'DocIDs': doc_ids_year, 'Num_Docs': num_docs,
                          'TermMap': term_map, 'Occurrences': occurrences}
                records.append(record)
        # Sort the records by the number of document
        # Write the output as a file
        df = pd.DataFrame(records, columns=['Collocation', 'Score', 'Num_Docs', 'DocIDs', 'TermMap', 'Occurrences'])
        # df = df.reset_index()
        path = os.path.join('output', self.args.case_name + '_collocations.csv')
        df.to_csv(path, encoding='utf-8', index=False)
        # # Write to a json file
        path = os.path.join('output', self.args.case_name + '_collocations.json')
        df.to_json(path, orient='records')
        print('Output keywords/phrases to ' + path)


# Main entry
if __name__ == '__main__':
    termGenerator = TermGenerator()
    # termGenerator.collect_terms_from_TFIDF()
    # termGenerator.collect_term_frequency()
    termGenerator.collect_collocation_doc_ids()


