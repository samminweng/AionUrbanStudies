'use strict';
const corpus = 'UrbanStudyCorpus';
const collocation_file_path = 'data/' + corpus + '_collocations.json';
const occurrence_file_path = 'data/' + corpus + '_occurrences.json';
// Document ready event
$(function () {
    // Load collocations
    $.when(
        $.getJSON(collocation_file_path), $.getJSON(occurrence_file_path)
    ).done(function (result1, result2){
        const collocation_data = result1[0];
        const occurrence_data = result2[0];
        // console.log(collocation_data);
        let network_chart = new NetworkChart(collocation_data, occurrence_data);
    });
});
