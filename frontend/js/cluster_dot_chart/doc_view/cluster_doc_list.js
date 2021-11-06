function ClusterDocList(cluster, doc_data) {
    const cluster_no = cluster['Cluster'];
    const cluster_topics = cluster['TopicN-gram'].slice(0, 10);
    const cluster_docs = doc_data.filter(d => cluster['DocIds'].includes(parseInt(d['DocId'])));
    console.log(cluster);
    console.log(cluster_docs);
    // // Obtain the articles containing the topics if topic is not null. Otherwise, obtain all the articles
    // const topic_docs = topic !== null ? cluster_docs) : cluster_docs;

    function _createUI() {
        $('#cluster_doc_list').empty();
        // Create a div to display
        const heading_text = 'Cluster #' + cluster_no + ' has ' + cluster_docs.length + ' articles ';
        const heading_div = $('<div class="h5">' + heading_text + '</div>')
        $('#cluster_doc_list').append(heading_div);

        // Create a div to display a list of topic (a link)
        const topic_text = cluster_topics.map(topic => topic['topic'] + ' (' + topic['doc_ids'].length + ')').join(" ");
        const available_topic_text = cluster['TopicN-gram'].slice(0, 30)
            .map(topic => topic['topic'] + ' (' + topic['doc_ids'].length + ')').join(" ");
        const topic_div = $('<div class="m-3"><h3><span class="fw-bold">Top 10 topics: </span>' + topic_text + '</h3>' +
            '<div><p>' + available_topic_text + '</p></div>' +
            '</div>');
        topic_div.accordion({
            icons: null,
            collapsible: true,
            heightStyle: "fill",
            active: 2
        });
        $('#cluster_doc_list').append(topic_div);

        // // Add 'reset' button
        // if (topic !== null) {
        //     const reset_btn = $('<button><span class="ui-icon ui-icon-closethick"></span></button>');
        //     reset_btn.button();
        //     reset_btn.click(function (event) {
        //         // // Get the documents about the topic
        //         const doc_list_heading = new DocumentListHeading(cluster_topics, cluster_docs, null);
        //     });
        //     heading.find('.search_term').append(reset_btn);
        // }
        // container.append($('<div class="row p-3"><div class="col"></div></div>').find(".col").append(heading));
        // $('#document_list_heading').append(container);

        // Create doc list
        const doc_list = new DocList(cluster_docs, cluster_topics);
    }

    _createUI();
}
