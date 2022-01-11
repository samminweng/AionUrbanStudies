'use strict';
const corpus = 'CultureUrbanStudyCorpus';
const cluster_approach = "HDBSCAN";

// Document ready event
$(function () {
    const progress_bar = new ProgressBar();
    // Clustering chart data
    const cluster_chart_data_file_path = 'data/doc_cluster/' + corpus + '_clusters.json';
    // Document (article abstract and title)
    const corpus_file_path = 'data/' + corpus + '_cleaned.json';
    // HDBSCAN cluster and topic data
    const cluster_topic_key_phrase_file_path = 'data/doc_cluster/' + corpus + '_HDBSCAN_Cluster_topic_key_phrases.json';
    // Document key phrase data
    const key_phrase_file_path = 'data/doc_cluster/' + corpus + '_doc_key_phrases.json';
    // Load data
    $.when(
        $.getJSON(cluster_chart_data_file_path),
        $.getJSON(corpus_file_path),
        $.getJSON(cluster_topic_key_phrase_file_path),
        $.getJSON(key_phrase_file_path)
    ).done(function (result1, result2, result3, result4) {
        const cluster_chart_data = result1[0];
        const corpus_data = result2[0];
        const cluster_topic_key_phrases = result3[0];
        const corpus_key_phrases = result4[0];
        const is_hide = true;
        // Draw the chart and list the clusters/topic words
        const chart = new ScatterGraph(is_hide, cluster_chart_data, cluster_topic_key_phrases, corpus_data, corpus_key_phrases);
        // Add event to hide or show outlier
        $("#hide_outliers").checkboxradio({});
        $('#hide_outliers').bind("change", function () {
            const is_hide = $("#hide_outliers").is(':checked');
            const chart_doc_view = new ScatterGraph(is_hide, cluster_chart_data, cluster_topic_key_phrases, corpus_data, corpus_key_phrases);
        });
        const dialog = new InstructionDialog(false);
        // Remove the progress bar
        $('#progressbar').remove();
    });

});
