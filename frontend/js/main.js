'use strict';
const corpus = 'UrbanStudyCorpus';
const corpus_file_path = 'data/' + corpus + '.json';
const collocation_file_path = 'data/' + corpus + '_collocations.json';
const occurrence_file_path = 'data/' + corpus + '_occurrences.json';
// Document ready event
$(function () {
    // Load collocations
    $.when(
        $.getJSON(corpus_file_path), $.getJSON(collocation_file_path), $.getJSON(occurrence_file_path)
    ).done(function (result1, result2, result3){
        const corpus_data = result1[0];
        const collocation_data = result2[0];
        const occurrence_data = result3[0];
        const starting_year = 0;
        // console.log(collocation_data);
        let network_chart = new NetworkChart(corpus_data, collocation_data, occurrence_data, starting_year);
    });
    // Add event to the year range
    $('#year_range').on('change', function(e){
        let value = e.target.value;
        let text = 'No limits';
        let starting_year = 0;
        if(value === "1") {
            text = "2010";
        }else if(value === "2"){
            text = "2015";
        }else if(value === "3"){
            $('#year_range').val('0');
            alert("Cannot select starting year > 2015");
        }
        $('#year_range_label').text(text);

        // console.log(value);
    })



});
