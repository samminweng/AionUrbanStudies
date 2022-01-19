'use strict';
const corpus = 'CultureUrbanStudyCorpus';
const params = new URLSearchParams(window.location.search);
// Load cluster data and display the results
function load_cluster_data_display_results(corpus_data, cluster_topic_key_phrases) {
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
    // // Set the cluster #2 as default cluster
    const topic_list_view = new DocTopicKeyPhraseListView(selected_cluster_no, cluster_topic_key_phrases, corpus_data);
    // Bind the change to cluster no
    $('#cluster_no').on('change', function (event) {
        const cluster_no = parseInt(this.value);
        const topic_list_view = new DocTopicKeyPhraseListView(cluster_no, cluster_topic_key_phrases, corpus_data);
    });
}

// Document ready event
$(function () {
    const progress_bar = new ProgressBar();
    // Clustering docs data (document, abstract and title)
    const corpus_data_file_path = 'data/cluster/' + corpus + '_clusters.json';
    // HDBSCAN cluster and topic data
    const cluster_topic_key_phrase_file_path = 'data/cluster/' + corpus + '_cluster_topic_key_phrases.json';
    $.when(
        $.getJSON(corpus_data_file_path), $.getJSON(cluster_topic_key_phrase_file_path)
    ).done(function (result1, result2) {
        const corpus_data = result1[0];
        const cluster_topic_key_phrases = result2[0];
        // Load the cluster and display the results
        load_cluster_data_display_results(corpus_data, cluster_topic_key_phrases);
        // Set up print /download as a pdf
        $('#download_as_pdf').button();
        $('#download_as_pdf').click(function (event) {
            window.print();
        });
        // Remove the progress bar
        $('#progressbar').remove();
        const total_papers = corpus_data.length;
        const outlier_count = cluster_topic_key_phrases.find(c => c['Cluster'] === -1)['NumDocs'];
        const outlier_percent = 100 * (outlier_count / total_papers);
        $('#total_papers').text(total_papers);
        $('#total_clusters').text(cluster_topic_key_phrases.length -1);
        $('#outlier_papers').text(outlier_count + ' (' + outlier_percent.toFixed() + '%)')
    })
});
