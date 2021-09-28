class Utility {
    // Store each individual
    static doc_key_terms = [];
    // Get the clustered documents
    static get_documents_by_doc_ids(doc_ids){
        // Get all the documents of a cluster
        return Utility.doc_key_terms.filter(d => doc_ids.includes(parseInt(d['DocId'])));
    }

    // Get the topic words of a cluster sorted by a rank approach
    static get_cluster_topic_words(total_clusters, cluster_no, rank){
        const dict = Utility.cluster_topic_words_dict[total_clusters];
        const cluster = dict.find(c => c['Cluster'] === cluster_no);
        const topic_words = cluster['Topic_Words_'+rank];
        return topic_words
    }

}
