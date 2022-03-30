// Draw the term chart and document list
function TermChart(cluster, cluster_docs){
    let keyword_groups = cluster['KeyPhrases'];
    // function update_topic_words_by_group(){
    //     const top_terms = cluster['TopTerms'];
    //     console.log(top_terms);
    //     for(const keyword_group of keyword_groups){
    //         console.log(keyword_group);
    //         let key_phrases = keyword_group['key-phrases'];
    //         let topic_words =keyword_group['topic_words'];
    //         // for(const word of topic_words){
    //         //
    //         // }
    //
    //     }
    // }

    // Create a cluster heading
    function createClusterHeading(){
        // Display Top 10 Distinct Terms and grouped key phrases
        const create_cluster_term_div = function(){
            // Create a div to display a list of topic (a link)
            const container = $('<div></div>');
            let cluster_terms = cluster['Terms'].slice(0, 10);
            // Sort the terms by its number of docs
            cluster_terms.sort((a, b) => b['doc_ids'].length - a['doc_ids'].length);
            // Add top 10 cluster terms (each term is a link)
            for (const selected_term of cluster_terms) {
                const link = $('<span type="button" class="btn btn-link"> #'
                    + selected_term['term'] + "</span>");
                // Click on the link to display the articles associated with topic
                link.click(function () {
                    // Get a list of docs in relation to the selected topic
                    const term_docs = cluster_docs.filter(d => selected_term['doc_ids'].includes(d['DocId']));
                    // Create a list of articles associated with topic
                    const doc_list = new DocList(term_docs, [selected_term['term']], selected_term['term']);
                    document.getElementById('doc_list').scrollIntoView({behavior: "smooth",
                        block: "nearest", inline: "nearest"});
                });
                container.append(link);
            }
            return container;
        };

        const cluster_no = cluster['Cluster'];
        const heading = $('<div class="p-3"><span class="h5">Article Cluster #' + cluster_no +'</span></div>');
        const cluster_link = $('<button type="button" class="btn btn-link" >' + cluster_docs.length + ' articles</button>');
        // Define onclick event cluster link
        cluster_link.click(function(){
            const doc_list = new DocList(cluster_docs, [], "");
            document.getElementById('doc_list').scrollIntoView({behavior: "smooth",
                block: "nearest", inline: "nearest"});
        })

        heading.append(cluster_link);
        heading.append(create_cluster_term_div());
        $('#cluster_heading').empty();
        $('#cluster_heading').append(heading);
    }


    // Main entry
    function createUI(){
        try{
            $('#term_occ_chart').empty();
            $('#sub_group').empty();
            $('#doc_list').empty();
            $('#key_phrase_chart').empty();
            // Display bar chart to show groups of key phrases as default graph
            const graph = new BarChart(keyword_groups, cluster, cluster_docs);
            createClusterHeading();
        }catch (error){
            console.error(error);
        }
    }
    createUI();
}
