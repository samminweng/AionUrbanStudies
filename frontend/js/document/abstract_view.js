// Create a title view
function TitleView(_document, collocation){
    const document = _document;
    let container = $('<div class="small"></div>');
    this.get_container = function(){
        return container;
    }

    function _createUI(){
        container.text(document['Title']);
        // Mark the collocations
        container.mark(collocation, {
        })
    }

    _createUI();
}
