// Create a title view
function TextView(document, key_terms){

    let container = $('<div class="small"></div>');
    this.get_container = function(){
        return container;
    }
    // Highlight key terms
    function mark_key_terms(div){
        for(let key_term of key_terms){
            const group = Utility.get_group_number(key_term);
            const mark_options ={
                "separateWordSearch": false,
                "accuracy": {
                    "value": "complementary",
                    "limiters": [",", ".", "'s", "/", ";"]
                },
                "className": "keyword-group-" + group,
            }
            div.mark(key_term, mark_options);
        }
        return div;


    }

    function _createUI(){
        // Add the title
        let title_div = $('<div></div>');
        title_div.append($('<span class="fw-bold">Title: </span><span>' + document['Title'] +'</span>'));
        // Mark the collocations on title div
        title_div = mark_key_terms(title_div);
        // title_div.mark(key_term_1, mark_options)
        container.append(title_div);
        // Add the abstract
        let abstract_div = $('<div></div>');
        abstract_div.append($('<span class="fw-bold">Abstract: </span><span>' + document['Abstract'] +'</span>'));
        abstract_div = mark_key_terms(abstract_div);
        // abstract_div.mark(key_term_1, mark_options);
        container.append(abstract_div);
        // Add the author keywords
        let author_keyword_div = $('<div></div>');
        author_keyword_div.append($('<span class="fw-bold">Author Keywords: </span><span>' + document['Author Keywords'] +'</span>'));
        author_keyword_div = mark_key_terms(author_keyword_div);
        // author_keyword_div.mark(key_term_1, mark_options);
        container.append(author_keyword_div);
    }

    _createUI();
}
