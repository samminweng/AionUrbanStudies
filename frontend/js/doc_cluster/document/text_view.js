// Create a title view
function TextView(document, topic_word) {
    let container = $('<div class="small"></div>');
    this.get_container = function () {
        return container;
    }

    // Highlight key terms
    function mark_key_terms(div) {
        const mark_options = {
            "separateWordSearch": false,
            "accuracy": {
                "value": "exactly",
                "limiters": [",", ".", "'s", "/", ";"]
            },
            "acrossElements": true,
            "ignorePunctuation": ":;.,-–—‒_(){}[]!'\"+=".split(""),
            "className": "topic_word"
        }
        div.mark(topic_word, mark_options);

        return div;
    }

    function _createUI() {
        // Add the title
        let title_div = $('<div></div>');
        title_div.append($('<span class="fw-bold">Title: </span><span>' + document['Title'] + '</span>'));
        // Mark the collocations on title div
        title_div = mark_key_terms(title_div);
        container.append(title_div);
        // Add the abstract
        let abstract_div = $('<div class="col"></div>');
        abstract_div.append($('<span class="fw-bold">Abstract: </span><span>' + document['Abstract'] + '</span>'));
        abstract_div = mark_key_terms(abstract_div);
        container.append(abstract_div);
    }

    _createUI();
}
