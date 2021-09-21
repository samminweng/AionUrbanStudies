class Utility {
    static cluster_topic_words_dict = {};   // The map between clustered documents to topic words
    static corpus_data = [];
    // Get the clustered documents
    static get_cluster_documents(total_clusters, cluster_no){
        const dict = Utility.cluster_topic_words_dict[total_clusters];
        const cluster = dict.find(c => c['Cluster'] === cluster_no);
        const doc_ids = cluster['DocIds'];
        // Get all the documents of a cluster
        return Utility.corpus_data.filter(d => doc_ids.includes(parseInt(d['DocId'])));
    }

    // Get the topic words of a cluster sorted by a rank approach
    static get_cluster_topic_words(total_clusters, cluster_no, rank){
        const dict = Utility.cluster_topic_words_dict[total_clusters];
        const cluster = dict.find(c => c['Cluster'] === cluster_no);
        const topic_words = cluster['Topic_words_'+rank];
        return topic_words
    }

}
