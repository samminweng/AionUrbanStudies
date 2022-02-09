function ClusterDocList(cluster_no, corpus_data, cluster_data) {
    const cluster = cluster_data.find(c => c['Cluster'] === cluster_no);
    const cluster_key_phrases = cluster['KeyPhrases'];
    const cluster_lda_topics = cluster['LDATopics'];
    const cluster_docs = corpus_data.filter(d => cluster['DocIds'].includes(parseInt(d['DocId'])));
    const cluster_link = $('<a target="_blank" href="term_sunburst.html?cluster='+ cluster_no + '">Cluster #' + cluster_no + '</a>');
    // const cluster_link = $('<span>Cluster #' + cluster_no + '</span>');
    if(cluster_no === -1){
        cluster_link.text("Outliers");
    }

    // Display Top 10 Distinct Terms and grouped key phrases
    function create_cluster_terms_key_phrase_topics(){
        // Create a div to display a list of topic (a link)
        $('#cluster_terms').empty();
        const container = $('<div><h5><span class="fw-bold">Distinct Terms by TF-IDF: </span></h5></div>');
        const term_p = $('<p></p>');
        let cluster_terms = cluster['Terms'].slice(0, 10);
        // Sort the terms by its number of docs
        cluster_terms.sort((a, b) => b['doc_ids'].length - a['doc_ids'].length);
        // Add top 10 cluster terms (each term is a link)
        for (const selected_term of cluster_terms) {
            const link = $('<button type="button" class="btn btn-link btn"> '
                + selected_term['term'] + ' (' + selected_term['doc_ids'].length + ')' + "</button>");
            // Click on the link to display the articles associated with topic
            link.click(function () {
                // Get a list of docs in relation to the selected topic
                const term_docs = cluster_docs.filter(d => selected_term['doc_ids'].includes(d['DocId']));
                // Create a list of articles associated with topic
                const doc_list = new DocList(term_docs, selected_term, cluster_data, null);
                document.getElementById('doc_list').scrollIntoView({behavior: "smooth",
                    block: "nearest", inline: "nearest"});
            });
            term_p.append(link);
        }

        // Append topic heading and paragraph to accordion
        container.append(term_p);
        $('#cluster_terms').append(container);

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
        $('#cluster_terms').append(accordion_div);
    }




    function _createUI() {
        // Create a div to display
        $('#cluster_doc_heading').empty();
        $('#cluster_doc_heading').append(cluster_link);
        const percent = parseInt(cluster['Percent'] * 100);
        $('#cluster_doc_heading').append($('<span> has ' +cluster_docs.length+ ' papers (' + percent + '%)</span>'));

        // Create a div to display top 10 Topic of a cluster
        create_cluster_terms_key_phrase_topics();

        // Create doc list
        const doc_list = new DocList(cluster_docs, null, null, null);
    }

    _createUI();
}
