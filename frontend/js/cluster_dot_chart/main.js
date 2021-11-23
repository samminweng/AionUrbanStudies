'use strict';
const corpus = 'UrbanStudyCorpus';

// Add the progress bar
function _createProgressBar(){
    // Update the progress bar asynchronously
    $('#progressbar').progressbar({
        value: 0,
        complete: function() {
            $( ".progress-label" ).text( "Complete!" );
        }
    });
    let counter = 0;
    (function asyncLoop() {
        $('#progressbar').progressbar("value", counter++);
        if (counter <= 100) {
            setTimeout(asyncLoop, 100);
        }
    })();
}

// Document ready event
$(function () {
    _createProgressBar();
    // Clustering chart data
    const cluster_chart_data_file_path = 'data/doc_cluster/' + corpus + '_clusters.json';
    // Document (article abstract and title)
    const corpus_file_path = 'data/' + corpus + '.json';
    // HDBSCAN cluster and topic words data
    const hdbscan_cluster_topic_words_file_path = 'data/doc_cluster/' + corpus + '_HDBSCAN_Cluster_topic_words.json';
    // Load data
    $.when(
        $.getJSON(cluster_chart_data_file_path),
        $.getJSON(corpus_file_path),
        $.getJSON(hdbscan_cluster_topic_words_file_path),
    ).done(function (result1, result2, result3) {
        const cluster_chart_data = result1[0];
        const corpus_data = result2[0];
        const cluster_topics = result3[0];
        const cluster_approach = "HDBSCAN";
        const is_hide = true;
        // Draw the chart and list the clusters/topic words
        const chart = new ScatterGraph(is_hide, cluster_chart_data, cluster_topics, corpus_data);
        // Add event to hide or show outlier
        $("#hide_outliers").checkboxradio({});
        $('#hide_outliers').bind("change", function () {
            const is_hide = $("#hide_outliers").is(':checked');
            const chart_doc_view = new ScatterGraph(is_hide, cluster_chart_data, cluster_topics, corpus_data);
        });
        const dialog = new InstructionDialog(true);
        // Remove the progress bar
        $('#progressbar').remove();
    });

});
