function DocumentListHeading(cluster_topics, cluster_docs, topic){
    const cluster_no = cluster_topics['Cluster'];
    // Obtain the articles containing the topics if topic is not null. Otherwise, obtain all the articles
    const topic_docs = topic !== null ? cluster_docs.filter(d => topic['doc_ids'].includes(parseInt(d['DocId']))) : cluster_docs;
    function _createUI(){
        $('#document_list_heading').empty();
        const container = $('<div class="container"></div>');
        const heading_text = 'Cluster #' + cluster_no + ' has ' + topic_docs.length + ' articles '
            + ((topic !== null)? "about <span class='search_term'> " + topic['topic'] + "</span>": "");
        // Display a summary on the heading
        const heading = $('<div class="h5 mb-3">' + heading_text + ' </div>');
        // Add 'reset' button
        if (topic !== null){
            const reset_btn = $('<button><span class="ui-icon ui-icon-closethick"></span></button>');
            reset_btn.button();
            reset_btn.click(function(event){
                // // Get the documents about the topic
                const doc_list_heading = new DocumentListHeading(cluster_topics, cluster_docs, null);
            });
            heading.find('.search_term').append(reset_btn);
        }
        container.append($('<div class="row p-3"><div class="col"></div></div>').find(".col").append(heading));
        $('#document_list_heading').append(container);

        // Create doc list
        const doc_list_view = new DocumentListView(cluster_topics, topic_docs, topic);
    }
    _createUI();
}
