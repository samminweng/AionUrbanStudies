'use strict';
const corpus = 'AIMLUrbanStudyCorpus';
const group_color_plates = {
    0: '#1f78b4',
    1: '#006d2c',
    2: '#fd8d3c',
    3: '#1a1a1a',
    4: '#e31a1c',
    5: '#6a3d9a'
    //3: Array.from(Array(13).keys()).map(i => d3.interpolateGreys(1-0.01*i)),
};

// Get the cluster color by group number
function get_color(abstract_cluster) {
    const cluster_no = abstract_cluster['cluster'];
    const group_no = abstract_cluster['cluster_group'];
    // Get the group colors < group_no
    let index = 0;
    for (let i = 0; i < group_no; i++) {
        index += group_color_plates[i].length;
    }
    // let color_index = cluster_no - index - 1;
    return group_color_plates[group_no];
}

// // Update the group number
function update_cluster_group_no(cluster_data){
    // cluster 12 - 16: group 3, cluster 17 - 21: group 4, cluster 22 - 24: group 5
    for (let i =0; i< cluster_data.length; i++){
        let cluster = cluster_data[i];
        if(cluster['cluster']>=12){
            if(cluster['cluster']<=16){
                cluster['cluster_group'] = 3
            }else if(cluster['cluster']<=21){
                cluster['cluster_group'] = 4
            }else if(cluster['cluster']<=24){
                cluster['cluster_group'] = 5
            }
        }
    }
    return cluster_data;
}

// Document ready event
$(function () {
    // Clustering docs data (document, abstract and title)
    const cluster_data_file_path = 'data/' + corpus + '_clusters.json';
    // HDBSCAN cluster and topic data
    const cluster_topic_key_phrase_file_path = 'data/' + corpus + '_cluster_terms_keyword_groups.json';
    // Load data
    $.when(
        $.getJSON(cluster_data_file_path),
        $.getJSON(cluster_topic_key_phrase_file_path)
    ).done(function (result1, result2) {
        const corpus_data = result1[0];
        const cluster_data = result2[0];
        // const cluster_data = update_cluster_group_no(result2[0]);
        // Draw the chart and list the clusters/topic words
        const chart = new ScatterGraph(corpus_data, cluster_data, null);
        // Update basic info
        const total_papers = corpus_data.length;
        $('#total_papers').text(total_papers);
    });

});
