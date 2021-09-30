// Create a table to show
function TopicListView(cluster_topics){
    const tf_idf_topics = cluster_topics['TopicWords_by_TF-IDF'].map(t => t['topic_words']);
    const topic2vec_topics = cluster_topics['TopicWords_by_CTF-IDF'].map(t => t['topic_words']);
    const collocation_topics = cluster_topics['TopicWords_by_Collocations'].map(t => t['topic_words']);

    function _createUI(){

        const key_term_div = $('<div><h3><span class="fw-bold">Topic by TF-IDF: </span></h3>' +
            '<div><p>' + tf_idf_topics.join("; ") + '</p></div>' +
            '</div>');
    }

    _createUI();
}
