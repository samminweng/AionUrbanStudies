'use strict';
// const corpus = 'CultureUrbanStudyCorpus';
const corpus = 'AIMLUrbanStudyCorpus';
const cluster_path = corpus + '_cluster_terms_key_phrases_LDA_topics.json';
const corpus_path = corpus + '_clusters.json';

const params = new URLSearchParams(window.location.search);
// Select cluster number
let selected_cluster_no = 0;
let selected_sub_cluster_no = 1;


// Display the sub-clusters of a large cluster
function displaySubCluster(sub_cluster_data) {
    // const sub_cluster_data = sub_cluster_dict[parent_cluster_no];
    const div = $("<div></div>");
    // Create a select
    const select_drop = $('<select></select>');
    // Add the options
    const sub_cluster_groups = sub_cluster_data['SubClusters'];
    sub_cluster_groups.sort((a, b) => b['DocIds'].length - a['DocIds'].length);
    const corpus = sub_cluster_data['Corpus'];
    for (let i = 0; i < sub_cluster_groups.length; i++) {
        const sub_cluster = sub_cluster_groups[i];
        // console.log(sub_cluster);
        const sub_cluster_no = sub_cluster['Cluster'];
        // Get the cluster terms
        const cluster_terms = sub_cluster['Terms'];
        const top_terms = get_top_terms(cluster_terms, 3);
        const cluster_docs = corpus.filter(d => sub_cluster['DocIds'].includes(d['DocId']));
        const option = $('<option value="' + sub_cluster_no + '"></option>');
        option.text(top_terms.join(", ") + ' (' + cluster_docs.length + ' papers)');
        select_drop.append(option);
        // Updated the Top Terms
        sub_cluster['TopTerms'] = top_terms;
    }
    select_drop.val(selected_sub_cluster_no);

    div.append($("<label>Select a sub-group: </label>"))
    div.append(select_drop);
    $('#sub_cluster_list').append(div);
    // Make a dropdown menu
    select_drop.selectmenu({
        width: 'auto',
        change: function (event, data) {
            // console.log( data.item.value);
            const sub_cluster_no = parseInt(data.item.value);
            const sub_cluster = sub_cluster_groups.find(c => c['Cluster'] === sub_cluster_no);
            const cluster_docs = corpus.filter(d => sub_cluster['DocIds'].includes(d['DocId']));
            const chart = new TermChart(sub_cluster, cluster_docs);
        }
    });
    // Create a term chart
    const sub_cluster = sub_cluster_groups.find(c => c['Cluster'] === selected_sub_cluster_no);
    const cluster_docs = corpus.filter(d => sub_cluster['DocIds'].includes(d['DocId']));
    const chart = new TermChart(sub_cluster, cluster_docs);
}

// Display the results of a cluster
function displayChartByCluster(cluster_no, clusters, corpus_data, sub_cluster_dict) {
    $('#sub_cluster_list').empty();
    const cluster_data = clusters.find(c => c['Cluster'] === cluster_no);
    // console.log(cluster_data);
    const cluster_docs = corpus_data.filter(d => cluster_data['DocIds'].includes(d['DocId']));
    const cluster_name = "cluster_" + cluster_no;
    if (cluster_name in sub_cluster_dict) {
        const sub_cluster_data = sub_cluster_dict[cluster_name];
        displaySubCluster(sub_cluster_data);
    } else {
        // Create a term chart
        const chart = new TermChart(cluster_data, cluster_docs);
    }
}

// Collect the unique top 3 terms
function get_top_terms(cluster_terms, n) {
    let top_terms = [];
    for (const cluster_term of cluster_terms) {
        const term = cluster_term['term'];
        // Check if the term overlap with any top term
        const overlap_top_term = top_terms.find(top_term => {
            // Split the terms into words and check if two term has overlapping word
            // For example, 'land use' is in the list. 'urban land' should not be included.
            const top_term_words = top_term.split(" ");
            const term_words = term.split(" ");
            if (top_term_words.length === term_words.length) {
                for (const top_term_word of top_term_words) {
                    for (const term_word of term_words) {
                        const over_lapping_word = top_term_words.find(t => t.toLowerCase() === term_word.toLowerCase());
                        if (over_lapping_word) {
                            return true;
                        }
                    }
                }
            }
            return false;
        })
        // The term does not overlap any existing top term
        if (!overlap_top_term) {
            // Check if any term word exist in top_terms
            const found_term = top_terms.find(top_term => {
                // Check if term is a part of an existing top term or an existing top term is part of the term
                // For example, 'cover' is part of 'land cover'
                if (term.toLowerCase().includes(top_term.toLowerCase()) ||
                    top_term.toLowerCase().includes(term.toLowerCase())) {
                    return true;
                }

                // Otherwise, return false
                return false;
            });
            if (found_term) {
                const found_index = top_terms.indexOf(found_term);
                // Replace
                top_terms[found_index] = term;
            } else {
                top_terms.push(term);
            }
            // Check if top_terms has three terms
            if (top_terms.length === n) {
                break;
            }
        }
    }
    const index = top_terms.findIndex(top_term => top_term === 'thing');
    // Replace thing to IoT
    if (index > 0) {
        top_terms[index] = "IoT";
    }
    return top_terms;
}


// Document ready event
$(function () {
    if (params.has('cluster')) {
        selected_cluster_no = parseInt(params.get('cluster'));
    }
    // Load collocations and tfidf key terms
    $.when(
        $.getJSON('data/' + cluster_path), $.getJSON('data/' + corpus_path),
        $.getJSON('data/cluster_-1/' + cluster_path), $.getJSON('data/cluster_-1/' + corpus_path),
        $.getJSON('data/cluster_0/' + cluster_path), $.getJSON('data/cluster_0/' + corpus_path),
        $.getJSON('data/cluster_1/' + cluster_path), $.getJSON('data/cluster_1/' + corpus_path),
        $.getJSON('data/cluster_2/' + cluster_path), $.getJSON('data/cluster_2/' + corpus_path),
        $.getJSON('data/cluster_3/' + cluster_path), $.getJSON('data/cluster_3/' + corpus_path)
    ).then()
        .done(function (result1, result2, result3, result4, result5, result6, result7, result8, result9, result10,
                        result11, result12) {
            let clusters = result1[0];
            const corpus_data = result2[0];
            const sub_cluster_dict = {
                "cluster_-1": {'SubClusters': result3[0], 'Corpus': result4[0]},
                "cluster_0": {'SubClusters': result5[0], 'Corpus': result6[0]},
                "cluster_1": {'SubClusters': result7[0], 'Corpus': result8[0]},
                "cluster_2": {'SubClusters': result9[0], 'Corpus': result10[0]},
                "cluster_3": {'SubClusters': result11[0], 'Corpus': result12[0]},
            };
            // Populate the top terms
            for (let cluster of clusters) {
                const cluster_terms = cluster['Terms'];
                // console.log(cluster_terms);
                const top_terms = get_top_terms(cluster_terms, 3);
                // console.log(top_terms);
                cluster['TopTerms'] = top_terms;
            }
            // Display a cluster as default cluster
            displayChartByCluster(selected_cluster_no, clusters, corpus_data, sub_cluster_dict);
            $("#cluster_list").empty();
            // Add a list of clusters/sub-clusters
            for (const cluster of clusters) {
                const cluster_no = cluster['Cluster'];
                const top_terms = cluster['TopTerms'];
                const cluster_doc_ids = cluster['DocIds'];
                if (cluster_no !== selected_cluster_no) {
                    $("#cluster_list").append($('<option value="' + cluster_no + '"> ' + top_terms.join(", ") +
                        ' (' + cluster_doc_ids.length + ' papers)  </option>'));
                } else {
                    $("#cluster_list").append($('<option value="' + cluster_no + '" selected> ' + top_terms.join(", ") +
                        ' (' + cluster_doc_ids.length + ' papers)  </option>'));
                }
            }
            $("#cluster_list").selectmenu({
                width: 'auto',
                change: function (event, data) {
                    // console.log( data.item.value);
                    const cluster_no = parseInt(data.item.value);
                    displayChartByCluster(cluster_no, clusters, corpus_data, sub_cluster_dict);
                }
            });

        });

})
