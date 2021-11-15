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
        const cluster_topic_words_data = result2[0];
        const cluster_sim_matrix = result3[0];
        const cluster_cluster_topics = result4[0];

        console.log(cluster_sim_matrix);

        // Display sunburst chart as default
        // const chart = new SunburstChart(cluster_groups, cluster_data, doc_data);
        // Display a chord chart

        // const dialog = new InstructionDialog('sunburst', true);
        // Remove the progress bar
        $('#progressbar').remove();
    });



})

// // Switch between sunburst chart and word tree chart
// $('#chart_type').selectmenu({
//     change: function( event, data ) {
//         const item = data.item.value;
//         console.log(item);
//         // Clear right panel
//         $('#topic_doc_heading').empty();
//         $('#topic_list').empty();
//         $('#topic_doc_list').empty();
//         // Display the chart on the left panel
//         if(item === 'wordtree'){
//             const chart = new WordTree(cluster_groups, cluster_data, doc_data);
//             // Create the instruction dialog
//             const dialog = new InstructionDialog('wordtree', false);
//         }else{
//             const chart = new SunburstChart(cluster_groups, cluster_data, doc_data);
//             // Create the instruction dialog
//             const dialog = new InstructionDialog('sunburst', false);
//         }
//     }
// });
