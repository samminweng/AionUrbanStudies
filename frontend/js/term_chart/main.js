'use strict';
const corpus = 'CultureUrbanStudyCorpus';
const cluster_path = 'data/' + corpus + '_cluster_terms_key_phrases_LDA_topics.json';
const corpus_path = 'data/' + corpus + '_clusters.json';
// Document ready event
$(function () {
    // Load collocations and tfidf key terms
    $.when(
        $.getJSON(cluster_path), $.getJSON(corpus_path)
    ).done(function (result1, result2) {
        const clusters = result1[0];
        const corpus_data = result2[0];
        const cluster_data = clusters.find(c => c['Cluster'] === 8);
        console.log(cluster_data);
        // Create a term chart
        const term_chart = new TermChart(cluster_data, corpus_data);
        // // Collect all collocations
        // const all_collocations = collocation_data.map(c => c['Collocation']);
        // // Fill in auto-complete input
        // $('#input_term').autocomplete({
        //     source: all_collocations,
        //     classes: {
        //         "ui-autocomplete": "highlight"
        //     },
        //     change: function(event, ui){
        //         let term_chart = new TermChart($('#input_term').val(), collocation_data, doc_term_data);
        //     }
        // });
        // // Bind the enter key event
        // $('#input_term').keypress(function(e){
        //     if(e.which === 13){
        //         let term_chart = new TermChart($('#input_term').val(), collocation_data, doc_term_data);
        //     }
        // });
        //
        // // Bind the onchange event to create a network graph
        // $('#searched_term').on('change', function(){
        //     let term_chart = new TermChart($('#searched_term').val(), collocation_data, doc_term_data);
        // });

    });

})
