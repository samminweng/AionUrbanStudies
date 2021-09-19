'use strict';
const corpus = 'UrbanStudyCorpus';
// Document ready event
$(function () {
    const doc_cluster_5_file_path = 'data/doc_cluster/' + corpus + '_5_simplified_cluster_doc.json';
    const doc_cluster_10_file_path = 'data/doc_cluster/' + corpus + '_10_simplified_cluster_doc.json';
    const doc_cluster_15_file_path = 'data/doc_cluster/' + corpus + '_15_simplified_cluster_doc.json';
    // Load the map between cluster and documents
    const cluster_word_5_file_path = 'data/doc_cluster/' + corpus + '_5_cluster_topic_words.json';
    const cluster_word_10_file_path = 'data/doc_cluster/' + corpus + '_10_cluster_topic_words.json';
    const cluster_word_15_file_path = 'data/doc_cluster/' + corpus + '_15_cluster_topic_words.json';

    // Load collocations
    $.when(
        $.getJSON(doc_cluster_5_file_path),$.getJSON(doc_cluster_10_file_path),$.getJSON(doc_cluster_15_file_path),
        $.getJSON(cluster_word_5_file_path),$.getJSON(cluster_word_10_file_path),$.getJSON(cluster_word_15_file_path)
    ).done(function (result1, result2, result3, result4, result5, result6){
        const doc_cluster_data_dict = {5: result1[0], 10: result2[0], 15: result3[0]};
        Utility.cluster_topic_words_dict = {5: result4[0], 10: result5[0], 15: result6[0]};
        console.log(Utility.cluster_topic_words_dict);
        let total_clusters = 5;
        // Load cluster data and render scatter plot
        let chart_doc_view = new ChartDocView(total_clusters, doc_cluster_data_dict[total_clusters]);
        $('#cluster').selectmenu({
            change: function (event, data){
                let total_clusters = data.item.value;
                let chart_doc_view = new ChartDocView(total_clusters, doc_cluster_data_dict[total_clusters]);
                // alert(cluster);
            }
        })


    });


    // Add event to cluster



});
