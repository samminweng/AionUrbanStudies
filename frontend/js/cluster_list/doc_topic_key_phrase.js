// Create an auto-complete
function DocTopicKeyPhraseListView(cluster_no, cluster_topic_key_phrases, corpus_data) {
    // Fill out the topic using TF-IDF
    const cluster_data = cluster_topic_key_phrases.find(c => c['Cluster'] === cluster_no);
    const available_topics = cluster_data['Topics'];
    // Get the articles in cluster #no
    const cluster_doc_ids = cluster_data['DocIds'];
    const cluster_docs = corpus_data.filter(d => cluster_doc_ids.includes(parseInt(d['DocId'])));

    function _createUI() {
        $('#topics').empty();   // Clean the topics
        // Initialise the auto-complete topic with all topics
        $("#topics").autocomplete({
            source: available_topics.map(t => t['topic'])
        });
        // Search topic within a cluster
        $('#search').button();
        $('#search').click(function (event) {
            const select_topic = $('#topics').val();
            const topic = available_topics.find(t => t['topic'] === select_topic);
            console.log(topic);
            const topic_docs = cluster_docs.filter(d => topic['doc_ids'].includes(parseInt(d['DocId'])));
            const doc_list = new DocList(cluster_data, topic_docs, topic);
        });
        // Clear button to clearn search input and display all the articles
        $('#clear').button();
        $('#clear').click(function (event) {
            $("#topics").val("");
            const doc_list = new DocList(cluster_data, cluster_docs, null);
        });

        const accordion_div = $('<div></div>');
        const topic_list_view = new ClusterTopicListView(cluster_data, cluster_docs, accordion_div);
        const key_phrase_list_view = new ClusterKeyPhrase(cluster_data, cluster_docs, accordion_div);
        accordion_div.accordion({
            collapsible: true,
            heightStyle: "fill",
            active: 0
        });
        $('#topic_list_view').empty();
        $('#topic_list_view').append($("<div class='row'><div class='col'></div></div>").find(".col").append(accordion_div));

        const doc_list = new DocList(cluster_data, cluster_docs, null);

    }

    _createUI();
}
