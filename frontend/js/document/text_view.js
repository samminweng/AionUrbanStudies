// Create a title view
function TextView(_document, collocation){
    const document = _document;
    const mark_options ={
        "separateWordSearch": false,
        "accuracy": {
            "value": "exactly",
            "limiters": [",", ".", "'s", "/"]
        },
        "className": "keyword",
    }
    let container = $('<div class="small"></div>');
    this.get_container = function(){
        return container;
    }

    function _createUI(){
        // Add the title
        let title_div = $('<div></div>');
        title_div.append($('<span class="fw-bold">Title: </span><span>' + document['Title'] +'</span>'));
        // Mark the collocations on title div
        title_div.mark(collocation, mark_options)
        container.append(title_div);
        // Add the abstract
        let abstract_div = $('<div></div>');
        abstract_div.append($('<span class="fw-bold">Abstract: </span><span>' + document['Abstract'] +'</span>'));
        abstract_div.mark(collocation, mark_options);
        container.append(abstract_div);
    }

    _createUI();
}
