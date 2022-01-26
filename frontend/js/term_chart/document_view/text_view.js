// Create a title view
function TextView(document, searched_term, complementary_terms) {
    // console.log(document);
    let key_terms = document['KeyTerms'];

    let container = $('<div></div>');
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
                "acrossElements": true,
                "ignorePunctuation": ":;.,-–—‒_(){}[]!'\"+=".split(""),
                "className": class_name
            }
            div.mark(term, mark_options);
        }
        return div;
    }

    function _createUI() {
        const highlight_terms = key_terms.slice(0, 5).filter(term => term !== searched_term && !complementary_terms.includes(term));
        console.log(highlight_terms);
        // Add the key term div
        let key_term_div = $('<div></div>');

        // Add Accordion
        let collapse_btn = $('<button class="accordion-button" type="button">' +
            key_terms.slice(0, 5).join(", ") +' ...</button>');
        // Add a div
        let collapse_div = $('<div class="collapse"><div class="card card-body">' +
            '<span class="fw-bold">All TF-IDF Terms: </span><span class="small">' + key_terms.join(", ") + '</span></div></div>');
        collapse_btn.on("click", function(){
            if(collapse_div.hasClass("show")){
                collapse_btn.removeClass("collapsed");
                collapse_div.attr("class", "collapse");
            }else{
                collapse_btn.addClass("collapsed");
                collapse_div.attr("class", "collapse show");
            }
        });
        key_term_div.append(collapse_btn);
        key_term_div.append(collapse_div);
        container.append(key_term_div);

        // Add the title
        let title_div = $('<div class="mt-3"></div>');
        title_div.append($('<span class="fw-bold">Title: </span><span>' + document['Title'] + '</span>'));
        // Mark the collocations on title div
        title_div = mark_key_terms(title_div, highlight_terms, 'key_term');
        title_div = mark_key_terms(title_div, [searched_term], 'search_term');
        title_div = mark_key_terms(title_div, complementary_terms, 'complementary_term');

        container.append(title_div);
        // Add the abstract
        let abstract_div = $('<div class="col"></div>');
        abstract_div.append($('<span class="fw-bold">Abstract: </span><span>' + document['Abstract'] + '</span>'));
        abstract_div = mark_key_terms(abstract_div, highlight_terms, 'key_term');
        abstract_div = mark_key_terms(abstract_div, [searched_term], 'search_term');
        abstract_div = mark_key_terms(abstract_div, complementary_terms, 'complementary_term');

        // abstract_div.mark(key_term_1, mark_options);
        container.append(abstract_div);
        // Add the author keywords
        let author_keyword_div = $('<div class="col"></div>');
        author_keyword_div.append($('<span class="fw-bold">Author Keywords: </span><span>' + document['AuthorKeywords'] + '</span>'));
        author_keyword_div = mark_key_terms(author_keyword_div, highlight_terms, 'key_term');
        author_keyword_div = mark_key_terms(author_keyword_div, [searched_term], 'search_term');
        author_keyword_div = mark_key_terms(author_keyword_div, complementary_terms, 'complementary_term');

        // author_keyword_div.mark(key_term_1, mark_options);
        container.append($('<div class="row"></div>').append(author_keyword_div));

    }

    _createUI();
}
