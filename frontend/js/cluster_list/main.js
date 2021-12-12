'use strict';
const corpus = 'UrbanStudyCorpus';
const params = new URLSearchParams(window.location.search);
const cluster_approach = 'HDBSCAN';
// Load cluster data and display the results
function load_cluster_data_display_results(cluster_topic_key_phrases, corpus_data, corpus_key_phrases) {
    // Select cluster number
    let selected_cluster_no = 4;
    if (params.has('cluster')) {
        selected_cluster_no = parseInt(params.get('cluster'));
    }
    // Populate the cluster list
    const cluster_no_list = cluster_topic_key_phrases.map(c => c['Cluster']);
    $('#cluster_no').empty();
    // Fill in the cluster no options
    for (const cluster_no of cluster_no_list) {
        const option = $('<option value=' + cluster_no + '>#' + cluster_no + '</option>');
        if (cluster_no === selected_cluster_no) {
            option.attr("selected", "selected");
        }
        $('#cluster_no').append(option);
    }
    // Set the cluster #2 as default cluster
    const topic_list_view = new DocTopicKeyPhraseListView(selected_cluster_no, cluster_topic_key_phrases, corpus_data, corpus_key_phrases);
    // Bind the change to cluster no
    $('#cluster_no').on('change', function (event) {
        const cluster_no = parseInt(this.value);
        const topic_list_view = new DocTopicKeyPhraseListView(cluster_no, cluster_topic_key_phrases, corpus_data, corpus_key_phrases);
    });
}

// Document ready event
$(function () {
    const progress_bar = new ProgressBar();
    // Document (article abstract and title) and key terms data
    const corpus_file_path = 'data/' + corpus + '_cleaned.json';
    // HDBSCAN cluster and topic words data
    const cluster_topic_key_phrases_file_path = 'data/doc_cluster/' + corpus + '_HDBSCAN_Cluster_topic_key_phrases.json';
    // Document key phrase data
    const key_phrase_file_path = 'data/doc_cluster/' + corpus + '_doc_key_phrases.json';
    $.when(
        $.getJSON(corpus_file_path), $.getJSON(cluster_topic_key_phrases_file_path),
        $.getJSON(key_phrase_file_path)
    ).done(function (result1, result2, result3) {
        const corpus_data = result1[0];
        const cluster_topic_key_phrases = result2[0];
        const corpus_key_phrases = result3[0];
        // Load the cluster and display the results
        load_cluster_data_display_results(cluster_topic_key_phrases, corpus_data, corpus_key_phrases);
        // Set up print /download as a pdf
        $('#download_as_pdf').button();
        $('#download_as_pdf').click(function (event) {
            window.print();
        });
        // Remove the progress bar
        $('#progressbar').remove();

    })
});
