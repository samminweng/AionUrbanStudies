'use strict';
const corpus = 'UrbanStudyCorpus';
// Document ready event
$(function () {
    // Clustering chart data
    const cluster_chart_data_file_path = 'data/doc_cluster/' + corpus + '_clusters.json';
    // Document (article abstract and title) and key terms data
    const doc_key_terms_file_path = 'data/doc_cluster/' + corpus + '_doc_terms.json';
    // KMeans cluster and topic words data
    const kmeans_cluster_topic_words_file_path = 'data/doc_cluster/' + corpus + '_KMeans_Cluster_topic_words.json';
    // HDBSCAN cluster and topic words data
    const hdbscan_cluster_topic_words_file_path = 'data/doc_cluster/' + corpus + '_HDBSCAN_Cluster_topic_words.json';
    // Load collocations
    $.when(
        $.getJSON(cluster_chart_data_file_path),
        $.getJSON(doc_key_terms_file_path),
        $.getJSON(kmeans_cluster_topic_words_file_path),
        $.getJSON(hdbscan_cluster_topic_words_file_path),
    ).done(function (result1, result2, result3, result4){
        const cluster_chart_data = result1[0];
        const doc_key_terms = result2[0];
        const cluster_topic_words_data = {"KMeans": result3[0], "HDBSCAN": result4[0]};
        const cluster_approach = "HDBSCAN";
        const is_hide = true;
        // Draw the chart and list the clusters/topic words
        const chart_doc_view = new ChartDocView(is_hide, cluster_approach, cluster_chart_data, cluster_topic_words_data);
        // Add event to the selection of clustering approach
        $('#cluster_approach').selectmenu({
            change: function (event, data){
                const cluster_approach = data.item.value;
                const is_hide = $("#hide_outliers").is(':checked');// Show/hide outliers
                const chart_doc_view = new ChartDocView(is_hide, cluster_approach, cluster_chart_data, cluster_topic_words_data);
            }
        });
        // Add event to hide or show outlier
        $("#hide_outliers").checkboxradio({});
        $('#hide_outliers').bind("change", function(){
            const is_hide = $("#hide_outliers").is(':checked');
            const chart_doc_view = new ChartDocView(is_hide, cluster_approach, cluster_chart_data, cluster_topic_words_data);
            // alert(is_hide);
        })
    });

});
