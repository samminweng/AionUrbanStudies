'use strict';
const corpus = 'UrbanStudyCorpus';

// Document ready event
$(function () {
    // Document (article abstract and title) and key terms data
    const doc_key_terms_file_path = 'data/doc_cluster/' + corpus + '_doc_terms.json';
    // HDBSCAN cluster and topic words data
    const hdbscan_cluster_topic_words_file_path = 'data/doc_cluster/' + corpus + '_HDBSCAN_Cluster_topic_words.json';
    $.when(
        $.getJSON(doc_key_terms_file_path), $.getJSON(hdbscan_cluster_topic_words_file_path)
    ).done(function (result1, result2) {
        const doc_term_data = result1[0];
        const cluster_topic_data = result2[0];
        // console.log(doc_key_terms);
        const chart = new WordTree(cluster_topic_data, doc_term_data);
    });



})
