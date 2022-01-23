function ClusterDocList(cluster_no, corpus_data, cluster_topic_key_phrases) {
    const cluster = cluster_topic_key_phrases.find(c => c['Cluster'] === cluster_no);
    const cluster_topics = cluster['Topics'].slice(0, 10);
    const cluster_key_phrases = cluster['KeyPhrases'];
    const cluster_lda_topics = cluster['LDATopics'];
    const cluster_docs = corpus_data.filter(d => cluster['DocIds'].includes(parseInt(d['DocId'])));
    const cluster_link = $('<a target="_blank" href="cluster_list.html?cluster='+ cluster_no + '">Cluster #' + cluster_no + '</a>');
    if(cluster_no === -1){
        cluster_link.text("Outliers");
    }

    // Display Top 10 Topics and grouped key phrases
    function create_cluster_topic_key_phrases(){
        // Create a div to display a list of topic (a link)
        $('#cluster_topics').empty();
        const topic_container = $('<div><h5><span class="fw-bold">Top 10 topics: </span></h5></div>');
        const topic_p = $('<p></p>');
        // Add top 30 topics (each topic as a link)
        for (const selected_topic of cluster_topics) {
            const link = $('<button type="button" class="btn btn-link btn"> '
                + selected_topic['topic'] + ' (' + selected_topic['doc_ids'].length + ')' + "</button>");
            // Click on the link to display the articles associated with topic
            link.click(function () {
                // Get a list of docs in relation to the selected topic
                const topic_docs = cluster_docs.filter(d => selected_topic['doc_ids'].includes(d['DocId']));
                // Create a list of articles associated with topic
                const doc_list = new DocList(topic_docs, selected_topic, cluster_topic_key_phrases, null);
                document.getElementById('doc_list').scrollIntoView({behavior: "smooth",
                    block: "nearest", inline: "nearest"});
            });
            topic_p.append(link);
        }

        // Append topic heading and paragraph to accordion
        topic_container.append(topic_p);
        $('#cluster_topics').append(topic_container);

        const accordion_div = $('<div></div>');
        // // Add the key phrases grouped by similarity
        const key_phrase_div = new ClusterKeyPhrase(cluster_key_phrases, cluster_docs, accordion_div);
        // Create LDA Accordion
        const lda_topic_div = new ClusterLDATopics(cluster_lda_topics, accordion_div);
        // // Set accordion
        accordion_div.accordion({
            // icons: null,
            collapsible: true,
            heightStyle: "fill",
            active: 0
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
        const doc_list = new DocList(cluster_docs, null, null, null);
    }

    _createUI();
}
