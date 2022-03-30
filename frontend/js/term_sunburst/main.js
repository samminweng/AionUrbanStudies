'use strict';
// const corpus = 'CultureUrbanStudyCorpus';
const corpus = 'AIMLUrbanStudyCorpus';
const cluster_path = corpus + '_cluster_terms_key_phrases_LDA_topics.json';
const corpus_path = corpus + '_clusters.json';

const params = new URLSearchParams(window.location.search);
let selected_cluster_no = 1;
// Document ready event
$(function () {
    // Load collocations and tfidf key terms
    $.when(
        $.getJSON('data/' + cluster_path), $.getJSON('data/' + corpus_path),
    ).then()
        .done(function (result1, result2) {
            const cluster_data = result1[0];
            const corpus_data = result2[0];
            // Initialise the list of cluster no
            $('#cluster_list').empty();
            // Add cluster cluster list
            for (const cluster of cluster_data) {
                const cluster_no = cluster['Cluster'];
                const terms = Utility.get_top_terms(cluster['Terms'].map(t => t['term']), 3);
                // console.log(terms);
                const option = $('<option value="' + cluster_no + '"># ' + cluster_no + ' (' +
                                  terms.join(", ") + '...) </option>');
                $('#cluster_list').append(option);
            }
            // Set the default cluster no
            $('#cluster_list').val(selected_cluster_no);

            // // Define onclick event of cluster no
            $("#cluster_list").change(function () {
                const cluster_no = parseInt($("#cluster_list").val());
                if(cluster_no){
                    const cluster = cluster_data.find(c => c['Cluster'] === cluster_no);
                    const cluster_docs = corpus_data.filter(d => cluster['DocIds'].includes(d['DocId']));
                    const chart = new TermChart(cluster, cluster_docs);
                }
            });
            const cluster = cluster_data[0];
            const cluster_docs = corpus_data.filter(d => cluster['DocIds'].includes(d['DocId']));
            const chart = new TermChart(cluster, cluster_docs);

        });

})
