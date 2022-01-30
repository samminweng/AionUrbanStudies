'use strict';
// const corpus = 'CultureUrbanStudyCorpus';
const corpus = 'MLUrbanStudyCorpus';

// Document ready event
$(function () {
    const progress_bar = new ProgressBar();
    // Clustering docs data (document, abstract and title)
    const cluster_data_file_path = 'data/' + corpus + '_clusters.json';
    // HDBSCAN cluster and topic data
    const cluster_topic_key_phrase_file_path = 'data/' + corpus + '_cluster_terms_key_phrases_LDA_topics.json';
    // Load data
    $.when(
        $.getJSON(cluster_data_file_path),
        $.getJSON(cluster_topic_key_phrase_file_path)
    ).done(function (result1, result2) {
        const corpus_data = result1[0];
        const cluster_topic_key_phrases = result2[0];
        const is_hide = false;
        // Draw the chart and list the clusters/topic words
        const chart = new ScatterGraph(is_hide, corpus_data, cluster_topic_key_phrases);
        // // Add event to hide or show outlier
        $("#hide_outliers").checkboxradio({});
        $('#hide_outliers').bind("change", function () {
            const is_hide = $("#hide_outliers").is(':checked');
            const chart = new ScatterGraph(is_hide,  corpus_data, cluster_topic_key_phrases);
        });
        const dialog = new InstructionDialog(false);
        // Remove the progress bar
        $('#progressbar').remove();
        // Update basic info
        const total_papers = corpus_data.length;
        const outlier_count = cluster_topic_key_phrases.find(c => c['Cluster'] === -1)['NumDocs'];
        const outlier_percent = 100 * (outlier_count / total_papers);
        $('#total_papers').text(total_papers);
        $('#total_clusters').text(cluster_topic_key_phrases.length -1);
        $('#outlier_papers').text(outlier_count + ' (' + outlier_percent.toFixed() + '%)')

    });

});
