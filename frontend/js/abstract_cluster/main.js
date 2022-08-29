'use strict';
const corpus = 'AIMLUrbanStudyCorpus';
const group_color_plates = {
    0: Array.from(Array(5).keys()).map(i => d3.interpolateBlues(1-0.1*i)),
    1: Array.from(Array(3).keys()).map(i => d3.interpolateGreens(1-0.1*i)),
    2: ['#fd8d3c', '#feb24c', '#fed976'],
    3: Array.from(Array(13).keys()).map(i => d3.interpolateGreys(1-0.03*i)),
};
// Get the cluster color by group number
function get_color(abstract_cluster) {
    const cluster_no = abstract_cluster['cluster'];
    const group_no = abstract_cluster['cluster_group'];
    // Get the group colors < group_no
    let index = 0;
    for (let i = 0; i < group_no; i++) {
        index += group_color_plates[i].length;
    }
    let color_index = cluster_no - index - 1;
    return group_color_plates[group_no][color_index];
}

// Document ready event
$(function () {
    // Clustering docs data (document, abstract and title)
    const cluster_data_file_path = 'data/' + corpus + '_clusters.json';
    // HDBSCAN cluster and topic data
    const cluster_topic_key_phrase_file_path = 'data/' + corpus + '_cluster_terms_keyword_groups.json';
    // Load data
    $.when(
        $.getJSON(cluster_data_file_path),
        $.getJSON(cluster_topic_key_phrase_file_path)
    ).done(function (result1, result2) {
        const corpus_data = result1[0];
        const cluster_data = result2[0];
        // Draw the chart and list the clusters/topic words
        const chart = new ScatterGraph(corpus_data, cluster_data, null);
        // Update basic info
        const total_papers = corpus_data.length;
        $('#total_papers').text(total_papers);
    });

});
