class Utility {
    static cluster_topic_words_dict = {};   // The map between clustered documents to topic words
    static corpus_data = [];
    // Get the clustered documents
    static get_cluster_documents(total_clusters, cluster_no){
        const dict = Utility.cluster_topic_words_dict[total_clusters];
        const cluster = dict.find(c => c['Cluster'] === cluster_no);
        const doc_ids = cluster['DocIds'];
        const topic_words = cluster['TopWords'];
        // console.log(topic_words);
        const documents = Utility.corpus_data.filter(d => doc_ids.includes(parseInt(d['DocId'])));
        return {topic_words: topic_words, documents: documents};
    }

    // Get the annotation
    static produce_annotations(){

    }

}
