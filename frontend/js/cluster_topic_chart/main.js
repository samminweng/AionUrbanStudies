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
    // Document (article abstract and title) and key terms data
    const doc_file_path = 'data/doc_cluster/' + corpus + '_doc_terms.json';
    // HDBSCAN cluster and topic words data
    const cluster_topic_words_file_path = 'data/doc_cluster/' + corpus + '_HDBSCAN_Cluster_topic_words.json';
    // Get cluster similarity matrix
    const cluster_similarity_file_path = 'data/similarity/' + corpus + '_HDBSCAN_cluster_vector_similarity.json';
    // Get cluster top 10 topics
    const cluster_topics_file_path = 'data/similarity/' + corpus + '_HDBSCAN_topic_vectors.json';

    $.when(
        $.getJSON(doc_file_path), $.getJSON(cluster_topic_words_file_path), $.getJSON(cluster_similarity_file_path),
        $.getJSON(cluster_topics_file_path)
    ).done(function (result1, result2, result3, result4) {
        const doc_data = result1[0];
        const cluster_topic_words = result2[0];
        const cluster_sim_data = result3[0];
        const cluster_topics = result4[0];
        // Display a chord chart
        const chart = new ChordChart(cluster_sim_data, cluster_topics, cluster_topic_words, doc_data);
        // const dialog = new InstructionDialog('sunburst', true);
        // Remove the progress bar
        $('#progressbar').remove();
    });



})

