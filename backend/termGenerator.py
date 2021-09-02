import os.path
from argparse import Namespace
import pandas as pd
from nltk import BigramCollocationFinder, word_tokenize
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from Utility import Utility
import nltk


# Download all the necessary NLTK data
# nltk.download('stopwords')
# nltk.download('punkt')
# nltk.download('wordnet')
# nltk.download('averaged_perceptron_tagger')
# nltk.download('words')

class TermGenerator:
    def __init__(self):
        self.args = Namespace(
            case_name='UrbanStudyCorpus',
            path='data'
        )
        self.stopwords = list(stopwords.words('english'))
        self.stopwords.append("also")  # add extra stop words
        # self.lemmatizer = WordNetLemmatizer()

    # Extract important words from scikit TFIDF
    def collect_terms_from_TFIDF(self):
        path = os.path.join('data', self.args.case_name + '.csv')
        text_df = pd.read_csv(path)
        # Create the vectorizers
        records = list()
        corpus = list()
        # Search all the subject words
        for i, text in text_df.iterrows():
            try:
                document = Utility.create_document(text)
                corpus.append(document)
                record = {'DocId': text['DocId'], 'Cited by': text['Cited by'], 'Title': text['Title'],
                          'Abstract': text['Abstract'],
                          'Author Keywords': Utility.get_author_keywords(text['Author Keywords'])}
                # record['Key 1-Words'] = Utility.extract_terms_from_TFIDF(word_vectorizer, document, self.stopwords)
                # record['Key 2-Words'] = Utility.extract_terms_from_TFIDF(bigram_vectorizer, document, self.stopwords)
                # record['Key 3-Words'] = Utility.extract_terms_from_TFIDF(trigram_vectorizer, document, self.stopwords)
                records.append(record)
            except Exception as err:
                print("Error occurred! {err}".format(err=err))
        for n_gram_type in ['1-Words', '2-Words', '3-Words']:
            key_term_lists = Utility.extract_terms_from_TFIDF(n_gram_type, corpus, self.stopwords)
            for index, key_terms in enumerate(key_term_lists):
                records[index]['Key ' + n_gram_type] = key_terms
        # # Write the output to a file
        df = pd.DataFrame(records, columns=["DocId", "Cited by", "Key 1-Words", "Key 2-Words", "Key 3-Words",
                                            "Author Keywords", "Title", "Abstract"])
        path = os.path.join('output', self.args.case_name + '_key_terms.csv')
        df.to_csv(path, encoding='utf-8', index=False)
        # # Write to a json file
        path = os.path.join('output', self.args.case_name + '_key_terms.json')
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
        # Remove all the stop words
        finder = BigramCollocationFinder.from_documents(documents)
        collocations = Utility.get_collocations(bigram_measures, finder, self.stopwords)
        max_length = len(collocations['Likelihood_ratio'])
        col_doc_dict = Utility.create_collocation_document_dict(collocations['Likelihood_ratio'], text_df)
        records = list()
        for i in range(max_length):
            record = {}
            for associate_measure in ['PMI', 'Chi_square', 'Likelihood_ratio']:
                collocation = collocations[associate_measure][i]['collocation']
                doc_ids = Utility.get_term_doc_ids(collocation, col_doc_dict)
                record['Collocation By ' + associate_measure] = collocation
                record['Score By ' + associate_measure] = collocations[associate_measure][i]['score']
                record['Document By ' + associate_measure] = len(doc_ids)
                record['DocIDs By ' + associate_measure] = doc_ids
            records.append(record)
        # Write the output to a file
        df = pd.DataFrame(records, columns=['Collocation By PMI', 'Score By PMI', 'Document By PMI',
                                            'Collocation By Chi_square', 'Score By Chi_square',
                                            'Document By Chi_square',
                                            'Collocation By Likelihood_ratio', 'Score By Likelihood_ratio',
                                            'Document By Likelihood_ratio', 'DocIDs By Likelihood_ratio'])
        # df = pd.DataFrame(records, columns=['Collocation', 'Score',
        #                                     'Document By Likelihood_ratio'])
        path = os.path.join('output', self.args.case_name + '_collocations.csv')
        df.to_csv(path, encoding='utf-8', index=False)
        # # Write to a json file
        path = os.path.join('output', self.args.case_name + '_collocations.json')
        df.to_json(path, orient='records')
        print('Output keywords/phrases to ' + path)


# Main entry
if __name__ == '__main__':
    termGenerator = TermGenerator()
    termGenerator.collect_terms_from_TFIDF()
    # termGenerator.collect_term_frequency()
    # termGenerator.collect_and_rank_collocations()
