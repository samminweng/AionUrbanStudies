'use strict';
const corpus = 'CultureUrbanStudyCorpus';

// Document ready event
$(function () {
    const progress_bar = new ProgressBar();
    // Clustering docs data (document, abstract and title)
    const cluster_data_file_path = 'data/cluster/' + corpus + '_clusters.json';
    // HDBSCAN cluster and topic data
    const cluster_topic_key_phrase_file_path = 'data/cluster/' + corpus + '_cluster_topic_key_phrases.json';
    // Load data
    $.when(
        $.getJSON(cluster_data_file_path),
        $.getJSON(cluster_topic_key_phrase_file_path)
    ).done(function (result1, result2) {
        const cluster_data = result1[0];
        const cluster_topic_key_phrases = result2[0];
        const is_hide = false;
        // Draw the chart and list the clusters/topic words
        const chart = new ScatterGraph(is_hide, cluster_data, cluster_topic_key_phrases);
        // // Add event to hide or show outlier
        $("#hide_outliers").checkboxradio({});
        $('#hide_outliers').bind("change", function () {
            const is_hide = $("#hide_outliers").is(':checked');
            const chart = new ScatterGraph(is_hide,  cluster_data, cluster_topic_key_phrases);
        });
        const dialog = new InstructionDialog(false);
        // Remove the progress bar
        $('#progressbar').remove();
    });

});
