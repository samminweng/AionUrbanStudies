function YearControl(searched_term, term_map, occurrences, documents){
    let container = $('#year_control').attr('class', 'controls');

    function _createUI(){
        container.empty();
        //
        let div = $('<div class="force"></div>');
        div.append($('<p><label>Year</label>Shift the year to view the trend</p>'));
        // Label
        let label = $('<label for="year_range" class="form-label"></label>');
        // Output
        let year_output = $('<output>2021</output>');
        // Input range
        let year_range = $('<input type="range" min=0 max=3 step=1 value=3>');
        // Set onchange event
        year_range.on('change', function(e) {
            let value = e.target.value;
            let ending_year = 0;
            if (value === "0") {
                ending_year = 2010;
            } else if (value === "1") {
                ending_year = 2015;
            } else if (value === "2") {
                ending_year = 2018;
            } else if (value === "3") {
                ending_year = 2021;
            }
            year_output.text(ending_year);
            const filtered_documents = documents.filter(doc => doc['Year'] <= ending_year);
            // Get the filtered term map and occurrences
            const {filtered_term_map, filtered_occurrences} = TermChartUtility.filter_term_map(filtered_documents, term_map, occurrences);
            // Create a network work graph
            let network_graph = new D3NetworkGraph(searched_term, filtered_term_map, filtered_occurrences, filtered_documents);
        });

        label.append(year_output);
        label.append(year_range);
        div.append(label);
        container.append(div);
    }

    _createUI();
}
