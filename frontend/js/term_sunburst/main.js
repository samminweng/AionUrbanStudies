'use strict';
// const corpus = 'CultureUrbanStudyCorpus';
const corpus = 'MLUrbanStudyCorpus';
const cluster_path = 'data/' + corpus + '_cluster_terms_key_phrases_LDA_topics.json';
const corpus_path = 'data/' + corpus + '_clusters.json';
const params = new URLSearchParams(window.location.search);

// Display the results of a cluster
function displayChartByCluster(cluster_no, clusters, corpus_data) {
    const cluster_data = clusters.find(c => c['Cluster'] === cluster_no);
    console.log(cluster_data);
    const cluster_docs = corpus_data.filter(d => cluster_data['DocIds'].includes(d['DocId']))
    // console.log(cluster_docs);
    // Create a term chart
    const chart = new TermSunburst(cluster_data, cluster_docs);
    $('#sub_group').empty();
    $('#doc_list').empty();
}

// Collect the unique top 3 terms
function get_top_terms(cluster_terms, n) {
    let top_terms = [];
    for (const cluster_term of cluster_terms) {
        const term = cluster_term['term'];
        // Check if the term overlap with any top term
        const overlap_top_term = top_terms.find(top_term => {
            // Split the terms into words and check if two term has overlapping word
            // For example, 'land use' is in the list. 'urban land' should not be included.
            const top_term_words = top_term.split(" ");
            const term_words = term.split(" ");
            if(top_term_words.length === term_words.length){
                for(const top_term_word of top_term_words){
                    for(const term_word of term_words){
                        const over_lapping_word = top_term_words.find(t => t.toLowerCase() === term_word.toLowerCase());
                        if(over_lapping_word){
                            return true;
                        }
                    }
                }
            }
            return false;
        })
        // The term does not overlap any existing top term
        if(!overlap_top_term){
            // Check if any term word exist in top_terms
            const found_term = top_terms.find(top_term => {
                // Check if term is a part of an existing top term or an existing top term is part of the term
                // For example, 'cover' is part of 'land cover'
                if(term.toLowerCase().includes(top_term.toLowerCase()) ||
                    top_term.toLowerCase().includes(term.toLowerCase())){
                    return true;
                }

                // Otherwise, return false
                return false;
            });
            if(found_term){
                const found_index = top_terms.indexOf(found_term);
                // Replace
                top_terms[found_index] = term;
            }else{
                top_terms.push(term);
            }
            // Check if top_terms has three terms
            if(top_terms.length === n){
                break;
            }
        }
    }
    return top_terms;
}


// Document ready event
$(function () {
    // Select cluster number
    let selected_cluster_no = 3;
    if (params.has('cluster')) {
        selected_cluster_no = parseInt(params.get('cluster'));
    }

    // Load collocations and tfidf key terms
    $.when(
        $.getJSON(cluster_path), $.getJSON(corpus_path)
    ).done(function (result1, result2) {
        let clusters = result1[0];
        const corpus_data = result2[0];
        // Populate the top terms
        for(let cluster of clusters){
            const cluster_terms = cluster['Terms'];
            console.log(cluster_terms);
            const top_terms = get_top_terms(cluster_terms, 3);
            console.log(top_terms);
            cluster['TopTerms'] = top_terms;
        }

        displayChartByCluster(selected_cluster_no, clusters, corpus_data);   // Display the cluster #8 as default cluster
        $("#cluster_list").empty();
        // Add a list of LDA topics
        for (const cluster of clusters) {
            const cluster_no = cluster['Cluster'];
            const top_terms = cluster['TopTerms'];
            const cluster_doc_ids = cluster['DocIds'];
            if (cluster_no !== selected_cluster_no) {
                $("#cluster_list").append($('<option value="' + cluster_no + '"> ' + top_terms.join(", ") +
                    ' (' + cluster_doc_ids.length+ ' papers)  </option>'));
            } else {
                $("#cluster_list").append($('<option value="' + cluster_no + '" selected> ' + top_terms.join(", ") +
                    ' (' + cluster_doc_ids.length+ ' papers)  </option>'));
            }
        }
        $("#cluster_list").selectmenu({
            change: function (event, data) {
                // console.log( data.item.value);
                const cluster_no = parseInt(data.item.value);
                displayChartByCluster(cluster_no, clusters, corpus_data);
            }
        });
    });

})
