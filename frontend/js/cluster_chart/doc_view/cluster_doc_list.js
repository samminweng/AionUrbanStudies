function ClusterDocList(cluster_no, corpus_data, cluster_data) {
    const cluster = cluster_data.find(c => c['Cluster'] === cluster_no);
    const cluster_docs = corpus_data.filter(d => cluster['DocIds'].includes(parseInt(d['DocId'])));
    // const cluster_link = $('<a target="_blank" href="term_sunburst.html?cluster='+ cluster_no + '">Article Cluster #' + cluster_no + '</a>');
    // Display Top 10 Distinct Terms and grouped key phrases
    function create_cluster_terms_key_phrase_topics(){
        // Create a div to display a list of topic (a link)
        $('#cluster_terms').empty();
        const container = $('<div></div>');
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
                const doc_list = new DocList(term_docs, cluster, selected_term['term']);
                // document.getElementById('doc_list').scrollIntoView({behavior: "smooth",
                //     block: "nearest", inline: "nearest"});
            });
            term_p.append(link);
        }

        // Append topic heading and paragraph to accordion
        container.append(term_p);
        $('#cluster_terms').append(container);
    }
    function _createUI() {
        // Create a div to display
        $('#cluster_doc_heading').empty();
        const heading = $('<div>Article Cluster #' + cluster_no +' </div>');
        const cluster_link = $('<button type="button" class="btn btn-link" >' + cluster_docs.length + ' articles</button>');
        heading.append(cluster_link);
        $('#cluster_doc_heading').append(heading);
        // Create a div to display top 10 Topic of a cluster
        create_cluster_terms_key_phrase_topics();
        // Create doc list
        const doc_list = new DocList(cluster_docs, cluster, null);
        // Define the cluster link
        cluster_link.click(function(){
            // Display all the articles
            const doc_list = new DocList(cluster_docs, cluster, null);
        });

    }
    _createUI();
}
