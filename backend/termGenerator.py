import json
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
                          'Abstract': text['Abstract']}
                records.append(record)
            except Exception as err:
                print("Error occurred! {err}".format(err=err))
        for n_gram_type in ['uni-gram', 'bi-gram', 'tri-gram']:
            key_term_lists = Utility.extract_terms_from_TFIDF(n_gram_type, corpus, self.stopwords)
            for index, key_terms in enumerate(key_term_lists):
                records[index]['Key_' + n_gram_type] = key_terms
        # # Write the output to a file
        df = pd.DataFrame(records, columns=["DocId", "Year", "Key_bi-gram", "Title", "Abstract"])
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
        # Score and rank the collocations
        associate_measures = ['Likelihood_ratio']  # ['PMI', 'Chi_square', 'Likelihood_ratio']
        # Remove all the stop words
        finder = BigramCollocationFinder.from_documents(documents)
        collocations = Utility.get_collocations(bigram_measures, finder, self.stopwords, associate_measures)
        records = list()
        for collocation in collocations['Likelihood_ratio']:
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
    def collect_relation_between_collocation_doc_ids(self):
        # Read the corpus
        path = os.path.join('data', self.args.case_name + '.csv')
        text_df = pd.read_csv(path)
        # Read collocation
        path = os.path.join('output', self.args.case_name + '_collocations_likelihood.json')
        col_df = pd.read_json(path)
        # max_length = len(col_df['Collocation'].tolist())
        collocations = col_df['Collocation'].tolist()
        scores = col_df['Score'].tolist()
        col_doc_dict = Utility.create_collocation_document(collocations, text_df)
        records = list()
        for index, collocation in enumerate(collocations):
            score = scores[index]
            doc_ids = col_doc_dict[collocation]
            doc_ids_year = Utility.group_doc_ids_by_year(text_df, doc_ids)
            record = {'Collocation': collocation, 'Score': score,
                      'DocIDs': doc_ids_year}
            records.append(record)
        records = records[:10]
        # Write the output as a file
        df = pd.DataFrame(records, columns=['Collocation', 'Score', 'DocIDs'])
        df = df.reset_index()
        path = os.path.join('output', self.args.case_name + '_collocations.csv')
        df.to_csv(path, encoding='utf-8', index=False)
        # # Write to a json file
        path = os.path.join('output', self.args.case_name + '_collocations.json')
        df.to_json(path, orient='records')
        print('Output keywords/phrases to ' + path)

    # Compute the co-occurrence of terms by looking the document ids. If two terms
    def compute_co_occurrence_terms(self):
        # Read collocations
        path = os.path.join('output', self.args.case_name + '_collocations.json')
        col_df = pd.read_json(path)
        max_length = 10
        # Get the collocations
        collocations = col_df['Collocation'][:max_length]
        records = list()
        # Filter the co-occurrences
        for starting_year in [0, 2010, 2015, 2020]:
            occ = list()
            for i in range(len(collocations)):
                col_i = col_df.query('index == {i}'.format(i=i)).iloc[0]
                occ_i = list()  # the occurrence of collocation 'i' with other collocations
                for j in range(len(collocations)):
                    if i == j:
                        occ_i.append([])  # No links
                    else:
                        col_j = col_df.query('index == {j}'.format(j=j)).iloc[0]
                        years = sorted(list(filter(lambda y: int(y) > starting_year, (col_i['DocIDs'].keys()))),
                                       reverse=True)
                        # Find the documents between collocation 'i' and collocation 'j'
                        for year in years:
                            if year not in col_j['DocIDs']:
                                occ_i.append([])
                            else:
                                doc_id_i = col_i['DocIDs'][year]
                                doc_id_j = col_j['DocIDs'][year]
                                doc_ids_ij = set(doc_id_i).intersection(set(doc_id_j))
                                doc_ids_ij = sorted(list(doc_ids_ij))
                                occ_i.append(doc_ids_ij)
                occ.append(occ_i)
            # Store the occurrence results as a json
            occ_json = {'starting_year': starting_year, 'occurrences': occ}
            records.append(occ_json)
        # Sort the records by starting year
        # Write the json to a file
        path = os.path.join('output', self.args.case_name + '_occurrences.json')
        with open(path, "w") as out_file:
            out_file.write(json.dumps(records, indent=4))
        print('Output the occurrences to ' + path)


# Main entry
if __name__ == '__main__':
    termGenerator = TermGenerator()
    # termGenerator.collect_terms_from_TFIDF()
    # termGenerator.collect_term_frequency()
    # termGenerator.collect_and_rank_collocations()
    # termGenerator.collect_relation_between_collocation_doc_ids()
    termGenerator.compute_co_occurrence_terms()
