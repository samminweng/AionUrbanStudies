function TopicList(topics, docs, src, target){

    // Create a doc list heading
    function createDocListHeading(selected_topics, selected_docs) {
        $('#topic_doc_list_heading').empty();
        // // Add a heading
        if (selected_topics) {
            const heading = $('<div class="h5">' + selected_docs.length + ' articles contains' +
                ' <span class="search_term">' + selected_topics['topic'] + '</span></div>');
            // Add the reset to heading
            const reset_btn = $('<button><span class="ui-icon ui-icon-closethick"></span></button>');
            reset_btn.button();
            // // Define reset button's click event
            reset_btn.click(function (event) {
                $('#topic_doc_list_heading').empty();
                // // Get the documents about the topic
                $('#topic_doc_list_heading').append($('<div class="h5">' + docs.length + ' articles ' + '</div>'));
                // Create a list of docs
                const doc_list = new DocList(docs, topics, null);
            });
            heading.find(".search_term").append(reset_btn);
            $('#topic_doc_list_heading').append(heading);
        } else {
            $('#topic_doc_list_heading').append($('<div class="h5">' + docs.length + ' articles ' + '</div>'));
        }

    }

    function _createUI(){
        $('#topic_list').empty();
        const topic_text = topics.map(topic => topic['topic'] + ' (' + topic['doc_ids'].length + ')').join(" ");
        const topic_div = $('<div>' +
            '<h3><span class="fw-bold"> Cluster #' +src + ' and Cluster #' + target + ' Topics: </span>'
            + topic_text + '</h3>' +
            '<div><p></p></div></div>');
        const topic_p = topic_div.find("p");
        for (const topic of topics) {
            const link = $('<button type="button" class="btn btn-link btn-sm"> '
                + topic['topic'] + ' (' + topic['doc_ids'].length + ')' + "</button>");
            // // // Click on the link to display the articles associated with topic
            link.click(function () {
                const selected_topic = topic;
                console.log(selected_topic);
                $('#cluster_doc_heading').append(
                    $('<span> has ' + selected_topic['doc_ids'].length + ' articles about '
                        + '<span class="search_term">' + selected_topic['topic'] + '</span></span>'));
                // // Get a list of docs in relation to the selected topic
                const topic_docs = docs.filter(d => selected_topic['doc_ids'].includes(d['DocId']));
                createDocListHeading(selected_topic, topic_docs);
                const doc_list = new DocList(topic_docs, topics, selected_topic);
            });
            topic_p.append(link);
        }
        // Set accordion
        topic_div.accordion({
            icons: null,
            collapsible: true,
            heightStyle: "fill",
            active: 2
        });

        $('#topic_list').append(topic_div);

        // Displays the topic list heading
        createDocListHeading(null);
        // Display all the articles relevant topics
        const doc_list = new DocList(docs, topics, null);


    }

    _createUI();
}
