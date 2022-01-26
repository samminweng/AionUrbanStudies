// A view displays the selected terms
function SelectedTermView(searched_term, documents){
    function _createUI(){
        let container = $('#selected_term');
        container.empty();
        // Add the selected terms
        let selected_term_div = $('<div class="col">' +
            '<label class="form-label">Selected key terms:</label>' +
            '<span id="selected_term" class="search_term p-2">' + searched_term + '</span>' +
            '</div>');
        // Complementary terms
        let complementary_term_span = $('<span id="complementary_term" class="complementary_term p-2"></span>');
        selected_term_div.append(complementary_term_span);
        // Clear button to clear complementary terms
        let button = $('<button type="button" class="btn btn-link">Clear</button>');
        button.on('click', function(){
            // Clear complementary terms
            complementary_term_span.empty();
            // Display the articles about 'searched_term'
            let doc_list_view = new DocumentListView([searched_term], [], documents);
        });

        selected_term_div.append(button);
        container.append($('<div class="row"></div>').append(selected_term_div));
    }

    _createUI();
}
