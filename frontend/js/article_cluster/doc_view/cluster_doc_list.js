function ClusterDocList(cluster_no, corpus_data, cluster_data, color) {
    const cluster = cluster_data.find(c => c['Cluster'] === cluster_no);
    const cluster_docs = corpus_data.filter(d => cluster['DocIds'].includes(parseInt(d['DocId'])));
    // Display Top 10 Distinct Terms and grouped key phrases
    function create_cluster_terms_key_phrase_topics(){
        // Create a div to display a list of topic (a link)
        $('#cluster_terms').empty();
        const container = $('<div class="container-sm"></div>');
        const term_p = $('<div></div>');
        let cluster_terms = cluster['Terms'].slice(0, 10);
        // Sort the terms by its number of docs
        cluster_terms.sort((a, b) => b['doc_ids'].length - a['doc_ids'].length);
        // Add top 10 cluster terms (each term is a link)
        for (let i = 0; i<3; i++) {
            const row = $('<div class="row"></div>');
            for(let j=0; j<4; j++){
                let index = i*4 + j;
                const col = $('<div class="col"></div>');
                if(index < cluster_terms.length){
                    const term = cluster_terms[index];
                    const link = $('<button type="button" class="btn btn-link btn">'
                        + term['term'] + ' (' + term['doc_ids'].length + ')' + "</button>");
                    // Click on the link to display the articles associated with topic
                    link.click(function () {
                        // Get a list of docs in relation to the selected topic
                        const term_docs = cluster_docs.filter(d => term['doc_ids'].includes(d['DocId']));
                        // Create a list of articles associated with topic
                        const doc_list = new DocList(term_docs, cluster, term['term']);
                    });
                    col.append(link);
                }else{
                    col.append($("<div></div>"));
                }
                row.append(col);
            }

            term_p.append(row);
        }
        // Append topic heading and paragraph to accordion
        container.append(term_p);
        $('#cluster_terms').append(container);
    }
    function _createUI() {
        // Create a div to display
        $('#cluster_doc_heading').empty();
        const score = cluster['Score'];
        const heading = $('<div>' +
            '<span style="color: ' +color +'">Article Cluster ' + cluster_no +'</span> ' +
            'has ' + cluster_docs.length + ' articles ' +
            'and <span class="score">' + score.toFixed(2) + '</span> score</div>');

        if(score < 0.0){
            heading.find(".score").addClass('text-danger');
        }
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
