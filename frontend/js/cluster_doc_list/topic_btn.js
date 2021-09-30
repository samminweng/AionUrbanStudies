// Create an auto-complete
function TopicBtn(cluster_no, cluster_topic_words, doc_key_terms){
    // Fill out the topic using TF-IDF
    const cluster_topics = cluster_topic_words.find(c => c['Cluster'] === cluster_no);
    const topic_words = cluster_topics['TopicWords_by_TF-IDF'];
    const available_topics = topic_words.map(t => t['topic_words']);
    const cluster_doc_ids = cluster_topics['DocIds'];
    const cluster_docs = doc_key_terms.filter(d => cluster_doc_ids.includes(parseInt(d['DocId'])));

    function _createUI(){
        $('#topics').empty();   // Clean the topics
        $( "#topics" ).autocomplete({
            source: available_topics
        });

        // Search topic within a cluster
        $('#search').button();
        $('#search').click(function(event){
            const select_topic = $('#topics').val();
            const topic = topic_words.find(t => t['topic_words'] === select_topic);
            // Get the articles related to the selected topic.
            const cluster_docs = doc_key_terms.filter(d => topic['doc_ids'].includes(parseInt(d['DocId'])));
            const doc_list_view = new DocumentListView(cluster_no, cluster_docs, [topic['topic_words']]);
        });
        // Clear button to clearn search input and display all the articles
        $('#clear').button();
        $('#clear').click(function(event){
            $( "#topics" ).val("")
            // Display all the articles in a cluster
            const doc_list_view = new DocumentListView(cluster_no, cluster_docs, []);
        });

        // Display all the articles in a cluster
        const doc_list_view = new DocumentListView(cluster_no, cluster_docs, []);
    }
    _createUI();
}
