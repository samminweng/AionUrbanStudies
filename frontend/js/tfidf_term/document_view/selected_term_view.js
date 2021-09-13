// A view displays the selected terms
function SelectedTermView(searched_term){
    function _createUI(){
        let container = $('#selected_term');
        container.empty();
        // Add the selected terms
        let selected_term_div = $('<div class="col">' +
            '<label class="form-label">Selected key terms:</label>' +
            '<span id="selected_term" class="search_term p-2">' + searched_term + '</span>' +
            '<span id="complementary_term"></span>' +
            '</div>');
        container.append($('<div class="row"></div>').append(selected_term_div));
    }

    _createUI();
}
