'use strict';
const corpus = 'AIMLUrbanStudyCorpus';
const group_color_plates = {
    1: Array.from(Array(7).keys()).map(i => d3.interpolateBlues(1-0.08*i)),
    2: Array.from(Array(7).keys()).map(i => d3.interpolateGreens(1-0.08*i)),
    3: ['#fd8d3c', '#feb24c', '#fed976'],
    4: Array.from(Array(15).keys()).map(i => d3.interpolateGreys(1-0.03*i)),

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
        // Update basic info
        const total_papers = corpus_data.length;
        $('#total_papers').text(total_papers);
    });

});
