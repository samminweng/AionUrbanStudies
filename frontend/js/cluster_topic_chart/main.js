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
    const cluster_topics_file_path = 'data/doc_cluster/' + corpus + '_HDBSCAN_Cluster_topic_words.json';
    // Get the cluster groups
    const cluster_group_file_path = 'data/doc_cluster/' + corpus + '_cluster_groups.json';
    $.when(
        $.getJSON(doc_file_path), $.getJSON(cluster_topics_file_path),
        $.getJSON(cluster_group_file_path)
    ).done(function (result1, result2, result3) {
        const doc_data = result1[0];
        const cluster_data = result2[0];
        const cluster_groups = result3[0];
        // Switch between sunburst chart and word tree chart
        $('#chart_type').selectmenu({
            change: function( event, data ) {
                const item = data.item.value;
                console.log(item);
                // Clear right panel
                $('#topic_doc_heading').empty();
                $('#topic_list').empty();
                $('#topic_doc_list').empty();
                // Display the chart on the left panel
                if(item === 'wordtree'){
                    const chart = new WordTree(cluster_groups, cluster_data, doc_data);
                    // Create the instruction dialog
                    const dialog = new InstructionDialog('wordtree', false);
                }else{
                    const chart = new SunburstChart(cluster_groups, cluster_data, doc_data);
                    // Create the instruction dialog
                    const dialog = new InstructionDialog('sunburst', false);
                }
            }
        });
        // Display sunburst chart as default
        const chart = new SunburstChart(cluster_groups, cluster_data, doc_data);
        const dialog = new InstructionDialog('sunburst', true);
        // Remove the progress bar
        $('#progressbar').remove();
    });



})
