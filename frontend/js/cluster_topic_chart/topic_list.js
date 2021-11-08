function TopicList(cluster_topics, cluster_docs){

    function _createUI(){
        $('#topic_list').empty();
        const topic_text = cluster_topics.slice(0, 10).map(topic => topic['topic'] + ' (' + topic['doc_ids'].length + ')').join(" ");
        const topic_div = $('<div>' +
            '<h3><span class="fw-bold">Top 10 topics: </span>' + topic_text + '</h3>' +
            '<div><p></p></div></div>');
        const topic_p = topic_div.find("p");
        for (const topic of cluster_topics) {
            const link = $('<button type="button" class="btn btn-link btn-sm"> '
                + topic['topic'] + ' (' + topic['doc_ids'].length + ')' + "</button>");
            // // Click on the link to display the articles associated with topic
            link.click(function () {
                const selected_topic = topic;
                console.log(selected_topic);
                $('#cluster_doc_heading').append(
                    $('<span> has ' + selected_topic['doc_ids'].length + ' articles about '
                        + '<span class="search_term">' + selected_topic['topic'] + '</span></span>'));


                // // Get a list of docs in relation to the selected topic
                const topic_docs = cluster_docs.filter(d => selected_topic['doc_ids'].includes(d['DocId']));
                const doc_list = new DocList(topic_docs, cluster_topics, selected_topic, cluster_docs);
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
    }

    _createUI();
}
