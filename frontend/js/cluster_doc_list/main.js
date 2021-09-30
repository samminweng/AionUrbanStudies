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
        const doc_key_terms = result1[0];
        const cluster_topic_words = result2[0];
        const cluster_no_list = cluster_topic_words.filter(c => c['Cluster'] >=0).map(c => c['Cluster']);
        // Fill in the cluster no options
        for(const cluster_no of cluster_no_list){
            const option = $('<option value=' +cluster_no+ '>#' + cluster_no + '</option>');
            if(cluster_no === 2){
                option.attr("selected", "selected");
            }
            $('#cluster_no').append(option);
        }
        // Change the cluster
        $('#cluster_no').selectmenu({
                change: function (event, data){
                    const cluster_no = parseInt($('#cluster_no').val());
                    const topic_btn = new TopicBtnListView(cluster_no, cluster_topic_words, doc_key_terms);
                }
        });
        // Set the cluster #2 as default cluster
        const topic_btn = new TopicBtnListView(2, cluster_topic_words, doc_key_terms);
    })
});
