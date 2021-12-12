function ClusterDocList(cluster, corpus_data, corpus_key_phrases) {
    const cluster_no = cluster['Cluster'];
    const cluster_topics = cluster['TF-IDF-Topics'].slice(0, 10);
    const cluster_key_phrases = cluster['Grouped_Key_Phrases'];
    const cluster_docs = corpus_data.filter(d => cluster['DocIds'].includes(parseInt(d['DocId'])));
    const cluster_link = $('<a target="_blank" href="cluster_list.html?cluster='+ cluster_no + '">Cluster #' + cluster_no + '</a>');

    // Create a Top 10 Topic region
    function create_cluster_topic_key_phrases(){
        // Create a div to display a list of topic (a link)
        $('#cluster_topics').empty();
        // const topic_text = cluster_topics.slice(0, 10).map(topic => topic['topic']).join("; ");
        const accordion_div = $('<div class="container"></div>');
        const topic_heading = $('<h3><span class="fw-bold">Top 10 topics: </span></h3>');
        const topic_p = $('<div><p></p></div>');
        // Add top 30 topics (each topic as a link)
        for (const selected_topic of cluster_topics) {
            const link = $('<button type="button" class="btn btn-link btn-sm"> '
                + selected_topic['topic'] + ' (' + selected_topic['doc_ids'].length + ')' + "</button>");
            // Click on the link to display the articles associated with topic
            link.click(function () {
                // Get a list of docs in relation to the selected topic
                const topic_docs = cluster_docs.filter(d => selected_topic['doc_ids'].includes(d['DocId']));
                // Create a list of articles associated with topic
                const doc_list = new DocList(topic_docs, selected_topic, corpus_key_phrases, null);
                document.getElementById('doc_list').scrollIntoView({behavior: "smooth",
                    block: "nearest", inline: "nearest"});
            });
            topic_p.append(link);
        }

        // Append topic heading and paragraph to accordion
        accordion_div.append(topic_heading);
        accordion_div.append(topic_p);
        // Add the key phrases grouped by similarity
        const key_phrase_div = new ClusterKeyPhrase(cluster_key_phrases, cluster_docs, corpus_key_phrases, accordion_div);
        // Set accordion
        accordion_div.accordion({
            // icons: null,
            collapsible: true,
            heightStyle: "fill",
            active: 1
        });
        $('#cluster_topics').append(accordion_div);
    }

    function _createUI() {
        // Create a div to display
        $('#cluster_doc_heading').empty();
        $('#cluster_doc_heading').append(cluster_link);
        $('#cluster_doc_heading').append($('<span> has ' +cluster_docs.length+ ' articles</span>'));

        // Create a div to display top 10 Topic of a cluster
        create_cluster_topic_key_phrases();

        // Create doc list
        const doc_list = new DocList(cluster_docs, null, corpus_key_phrases, null);
    }

    _createUI();
}
