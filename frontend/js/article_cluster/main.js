'use strict';
const corpus = 'AIMLUrbanStudyCorpus';
const group_color_plates = {
    1: d3.schemeBlues[9].reverse().slice(0, 7),
    2: d3.schemeGreys[5].reverse().slice(0, 3),
    3: d3.schemeGreens[9].reverse().slice(0, 7),
    4: d3.schemeOranges[5].reverse().slice(0, 3),
    5: d3.schemeReds[9].reverse().slice(0, 8),
    6: d3.schemePurples[5].reverse().slice(0, 3)
};
// Document ready event
$(function () {
    // Clustering docs data (document, abstract and title)
    const cluster_data_file_path = 'data/' + corpus + '_clusters_updated.json';
    // HDBSCAN cluster and topic data
    const cluster_topic_key_phrase_file_path = 'data/' + corpus + '_cluster_terms_key_phrases_topics_updated.json';
    // Load data
    $.when(
        $.getJSON(cluster_data_file_path),
        $.getJSON(cluster_topic_key_phrase_file_path)
    ).done(function (result1, result2) {
        const corpus_data = result1[0];
        const cluster_data = result2[0];
        // Draw the chart and list the clusters/topic words
        const chart = new ScatterGraph(corpus_data, cluster_data, null);
        const view = new ArticleClusterList(corpus_data, cluster_data);
        // Update basic info
        const total_papers = corpus_data.length;
        $('#total_papers').text(total_papers);
        // // Initialise the list of cluster no
        // $('#cluster_list').empty();
        // $('#cluster_list').append($('<option value="all" selected="selected">ALL</option>'))
        // // Add cluster cluster list
        // for (const cluster of cluster_data) {
        //     const cluster_no = cluster['Cluster'];
        //     // console.log(terms);
        //     const option = $('<option value="' + cluster_no + '">' + cluster_no + '</option>');
        //     $('#cluster_list').append(option);
        // }
        // // // Define onclick event of cluster no
        // $("#cluster_list").change(function () {
        //     const cluster_no = parseInt($("#cluster_list").val());
        //     if (cluster_no) {
        //         const chart = new ScatterGraph(corpus_data, cluster_data, cluster_no);
        //     } else {
        //         // Display all article clusters
        //         const chart = new ScatterGraph(corpus_data, cluster_data, null);
        //     }
        // });
        // $('#cluster_list').hide();
    });

});
