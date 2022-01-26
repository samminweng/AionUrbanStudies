// Create a range slider
// Ref: https://slawomir-zaziablo.github.io/range-slider/
function YearControl(searched_term, term_map, documents) {
    const year_ranges = {"0": 2000, "1": 2010, "2": 2015, "3": 2018, "4": 2021}

    function _createUI() {
        let handler = $('#custom-handle');
        $('#year_range').slider({
            value: 4,
            min: 0,
            max: 4,
            step: 1,
            create: function () {
                handler.text(year_ranges[$(this).slider("value")]);
            },
            slide: function (event, ui) {
                let ending_year = year_ranges[ui.value];
                handler.text(ending_year);
                const filtered_documents = documents.filter(doc => doc['Year'] <= ending_year);
                // Get the filtered term map and occurrences
                const filtered_term_map = TermChartUtility.filter_term_map(filtered_documents, term_map);
                // Create a network work graph
                let network_graph = new D3NetworkGraph(searched_term, filtered_term_map, filtered_documents);
            }
        });
        // // Set onchange event
        // $('#year_range').on('change', function(e) {
        //     let value = e.target.value;
        //     let ending_year = year_ranges[value];
        //     $('#year_output').text(ending_year);
        //     const filtered_documents = documents.filter(doc => doc['Year'] <= ending_year);
        //     // Get the filtered term map and occurrences
        //     const filtered_term_map = TermChartUtility.filter_term_map(filtered_documents, term_map);
        //     // Create a network work graph
        //     let network_graph = new D3NetworkGraph(searched_term, filtered_term_map, filtered_documents);
        // });
        // // Reset the year range
        // $('#year_range').val("4");
        // $('#year_output').text(2021);

    }

    _createUI();
}
