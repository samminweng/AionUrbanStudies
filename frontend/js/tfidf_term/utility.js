class Utility {
    // Collect all the document
    static collect_documents_by_doc_ids(collocation, doc_term_data){
        const doc_ids = Object.values(collocation['DocIDs']).flat();
        console.log(doc_ids);
        return doc_term_data.filter(doc_term => doc_ids.includes(doc_term['DocId']));
    }
    // Group the documents by the last 5 key terms (extracted by using TF-IDF method) that frequently
    // appear in other documents
    static group_documents_key_terms(documents){
        let term_map = new Map();
        const num_of_terms = 5;
        for(const doc of documents){
            let doc_id = doc['DocId'];
            let length = doc['KeyTerms'].length;
            // Obtain the first 5 and last 5 key terms
            // let key_terms = doc['KeyTerms'].slice(0, num_of_terms).concat(doc['KeyTerms'].slice(length-num_of_terms, length));
            let key_terms = doc['KeyTerms'].slice(length-num_of_terms, length);
            for(let key_term of key_terms){
                // Initialise the term map with a set to store all the doc_ids
                if(!term_map.has(key_term)){
                    term_map[key_term] = new Set();
                }
                term_map[key_term].add(doc_id);
            }
            // For debugging
            if(!doc['KeyTerms'].includes('public transportation')){
                //
                alert('Document id = ' + doc_id + ' does not have public transport');
            }

        }
        return term_map;
    }



}



