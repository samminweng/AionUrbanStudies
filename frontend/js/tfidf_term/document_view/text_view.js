// Create a title view
function TextView(document, searched_term, complementary_terms) {
    // console.log(document);
    const key_terms = document['KeyTerms'].filter(term => searched_term !== term && !complementary_terms.includes(term)).slice(0, 5);
    console.log(key_terms);
    let container = $('<div class="small"></div>');
    this.get_container = function () {
        return container;
    }

    // Highlight key terms
    function mark_key_terms(div, terms, class_name) {
        for (let term of terms) {
            // Marking option
            const mark_options = {
                "separateWordSearch": false,
                "accuracy": {
                    "value": "exactly",
                    "limiters": [",", ".", "'s", "/", ";"]
                },
                "ignorePunctuation": ":;.,-–—‒_(){}[]!'\"+=".split(""),
                "className": class_name
            }
            div.mark(term, mark_options);
        }
        return div;
    }

    function _createUI() {
        // Add the title
        let title_div = $('<div></div>');
        title_div.append($('<span class="fw-bold">Title: </span><span>' + document['Title'] + '</span>'));
        // Mark the collocations on title div
        title_div = mark_key_terms(title_div, [searched_term], 'search_term');
        title_div = mark_key_terms(title_div, complementary_terms, 'complementary_term');
        title_div = mark_key_terms(title_div, key_terms, 'key_term');
        // title_div.mark(key_term_1, mark_options)
        container.append(title_div);
        // Add the abstract
        let abstract_div = $('<div></div>');
        abstract_div.append($('<span class="fw-bold">Abstract: </span><span>' + document['Abstract'] + '</span>'));
        abstract_div = mark_key_terms(abstract_div, [searched_term], 'search_term');
        abstract_div = mark_key_terms(abstract_div, complementary_terms, 'complementary_term');
        abstract_div = mark_key_terms(abstract_div, key_terms, 'key_term');
        // abstract_div.mark(key_term_1, mark_options);
        container.append(abstract_div);
        // Add the author keywords
        let author_keyword_div = $('<div></div>');
        author_keyword_div.append($('<span class="fw-bold">Author Keywords: </span><span>' + document['AuthorKeywords'] + '</span>'));
        author_keyword_div = mark_key_terms(author_keyword_div, [searched_term], 'search_term');
        author_keyword_div = mark_key_terms(author_keyword_div, complementary_terms, 'complementary_term');
        author_keyword_div = mark_key_terms(author_keyword_div, key_terms, 'key_term');
        // author_keyword_div.mark(key_term_1, mark_options);
        container.append(author_keyword_div);
    }

    _createUI();
}
