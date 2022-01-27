'use strict';
const corpus = 'CultureUrbanStudyCorpus';
const cluster_path = 'data/' + corpus + '_cluster_terms_key_phrases_LDA_topics.json';
const corpus_path = 'data/' + corpus + '_clusters.json';
// Display the results of a cluster
function displayChartByCluster(cluster_no, clusters, corpus_data){
    const cluster_data = clusters.find(c => c['Cluster'] === cluster_no);
    console.log(cluster_data);
    const cluster_docs = corpus_data.filter(d => cluster_data['DocIds'].includes(d['DocId']))
    console.log(cluster_docs);
    $('#cluster_no').text(cluster_no);
    $('#doc_count').text(cluster_docs.length);
    // Create a term chart
    const term_chart = new TermChart(cluster_data, cluster_docs);
}



// Document ready event
$(function () {
    // Load collocations and tfidf key terms
    $.when(
        $.getJSON(cluster_path), $.getJSON(corpus_path)
    ).done(function (result1, result2) {
        const clusters = result1[0];
        const corpus_data = result2[0];
        displayChartByCluster(0, clusters, corpus_data);   // Display the cluster #8 as default cluster
        $("#cluster_list").empty();
        // Add a list of LDA topics
        for(let i=0; i< clusters.length; i++){
            const cluster_no = clusters[i]['Cluster'];
            if(cluster_no !== 0){
                $("#cluster_list").append($('<option value="' + cluster_no+'"> Cluster #' + cluster_no +' </option>'));
            }else{
                $("#cluster_list").append($('<option value="' + cluster_no+'" selected> Cluster #' + cluster_no +' </option>'));
            }
        }
        $( "#cluster_list" ).selectmenu({
            change: function( event, data ) {
                // console.log( data.item.value);
                const cluster_no = parseInt(data.item.value);
                displayChartByCluster(cluster_no, clusters, corpus_data);
            }
        });
    });

})
