'use strict';
const corpus = 'UrbanStudyCorpus';
const ui_name = 'tfidf_term';
const collocation_file_path = 'data/' + ui_name + '/' + corpus + '_collocations.json';
const doc_term_file_path = 'data/' + ui_name + '/' + corpus + '_doc_terms.json';
// Document ready event
$(function () {
    // Load collocations and tfidf key terms
    $.when(
        $.getJSON(collocation_file_path), $.getJSON(doc_term_file_path),
    ).done(function (result1, result2) {
        const collocation_data = result1[0];
        const doc_term_data = result2[0];
        const searched_term = $('#searched_term').val();
        // console.log(searched_term);
        let term_chart = new TermChart(searched_term, collocation_data, doc_term_data);

    });

})
