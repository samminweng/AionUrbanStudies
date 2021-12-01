function ClusterDocHeading(cluster_topics, cluster_docs, topic, corpus_key_phrases) {
    const cluster_no = cluster_topics['Cluster'];
    // Obtain the articles containing the topics if topic is not null. Otherwise, obtain all the articles
    const topic_docs = topic !== null ? cluster_docs.filter(d => topic['doc_ids'].includes(parseInt(d['DocId']))) : cluster_docs;

    function _createUI() {
        $('#doc_list_heading').empty();
        const container = $('<div class="container"></div>');
        let heading_text = 'Cluster #' + cluster_no;
        if (topic !== null) {
            heading_text += ' has ' + topic_docs.length + " out of " + cluster_docs.length + " articles " +
                " about <span class='search_term'> " + topic['topic'] + "</span>";
        } else {
            heading_text += ' has ' + topic_docs.length + ' articles ';
        }
        // Display a summary on the heading
        const heading = $('<div class="h5 mb-3">' + heading_text + ' </div>');
        // Add 'reset' button
        if (topic !== null) {
            const reset_btn = $('<button><span class="ui-icon ui-icon-closethick"></span></button>');
            reset_btn.button();
            reset_btn.click(function (event) {
                // Get the documents about the topic
                const heading = new ClusterDocHeading(cluster_topics, cluster_docs, null, corpus_key_phrases);
            });
            heading.find('.search_term').append(reset_btn);
        }
        container.append($('<div class="row p-3"><div class="col"></div></div>').find(".col").append(heading));
        $('#doc_list_heading').append(container);

        // Create doc list
        const doc_list = new DocList(cluster_topics, topic_docs, topic, corpus_key_phrases);
    }

    _createUI();
}
