'use strict';
const corpus = 'AIMLUrbanStudyCorpus';
const select_no = 22;
// D3 category color pallets
const color_plates = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b",
    "#e377c2", "#7f7f7f", "#bcbd22", "#17becf"];
// Document ready event
$(function () {
    // Clustering docs data (document, abstract and title)
    const cluster_data_file_path = 'data/' + corpus + '_clusters_updated.json';
    // HDBSCAN cluster and topic data
    const cluster_topic_key_phrase_file_path = 'data/' + corpus + '_cluster_terms_keyword_groups_updated.json';
    // Load data
    $.when(
        $.getJSON(cluster_data_file_path),
        $.getJSON(cluster_topic_key_phrase_file_path)
    ).done(function (result1, result2) {
        const corpus_data = result1[0];
        const cluster_data = result2[0];
        // Draw the chart and list the clusters/topic words
        const chart = new ScatterGraph(corpus_data, cluster_data, select_no);
        // Update basic info
        const total_papers = corpus_data.length;
        $('#total_papers').text(total_papers);
        // Initialise the list of cluster no
        $('#cluster_list').empty();
        // Add cluster cluster list
        for (const cluster of cluster_data) {
            const cluster_no = cluster['Cluster'];
            // console.log(terms);
            const option = $('<option value="' + cluster_no + '">' + cluster_no + '</option>');
            $('#cluster_list').append(option);
        }
        // // Define onclick event of cluster no
        $("#cluster_list").change(function () {
            const cluster_no = parseInt($("#cluster_list").val());
            if (cluster_no) {
                const chart = new ScatterGraph(corpus_data, cluster_data, cluster_no);
            } else {
                const chart = new ScatterGraph(corpus_data, cluster_data, null);
            }
        });

        $('#cluster_list').val(select_no);
    });

});
