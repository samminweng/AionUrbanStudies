'use strict';
// const corpus = 'CultureUrbanStudyCorpus';
const corpus = 'AIMLUrbanStudyCorpus';
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
        // Draw the chart and list the clusters/topic words
        const chart = new ScatterGraph(corpus_data, cluster_topic_key_phrases);
        const dialog = new InstructionDialog(false);
        // Remove the progress bar
        $('#progressbar').remove();
        // Update basic info
        const total_papers = corpus_data.length;
        $('#total_papers').text(total_papers);
    });

});
