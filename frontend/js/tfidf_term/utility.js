class Utility {
    // Collect all the document
    static collect_documents_by_doc_ids(collocation, doc_term_data){
        const doc_ids = Object.values(collocation['DocIDs']).flat();
        console.log(doc_ids);
        return doc_term_data.filter(doc_term => doc_ids.includes(doc_term['DocId']));
    }



}



