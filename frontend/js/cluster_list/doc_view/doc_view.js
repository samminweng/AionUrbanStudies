// Create a title view
function DocView(doc, selected_term) {
    const doc_key_phrases = doc['KeyPhrases'];
    let search_terms = [];
    if(selected_term != null){
        if('topic' in selected_term){
            search_terms = [selected_term['topic'], selected_term['plural']]; // Highlight singular and plural topics
        }
    }


    const container = $('<div></div>');
    this.get_container = function () {
        return container;
    }

    // Highlight key terms
    function mark_key_terms(div, terms, class_name) {
        // Check if the topic is not empty
        for (const term of terms) {
            // Mark the topic
            const mark_options = {
                "separateWordSearch": false,
                "accuracy": {
                    "value": "exactly",
                    "limiters": [",", ".", "'s", "/", ";", ":", '(', ')', '‘', '’', '%']
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
        // Add Key Phrase
        let key_phrase_div = $('<div class="card border-info">' +
            '<div class="card-body">' +
            '<p class="card-text"><span class="fw-bold">Key Phrases: </span>' + doc_key_phrases.join(", ") + '</p>' +
            '</div></div>');
        container.append(key_phrase_div);

        // Add the title
        let title_div = $('<div></div>');
        title_div.append($('<span class="fw-bold">Title: </span><span>' + doc['Title'] + '</span>'));
        // Mark the collocations on title div
        title_div = mark_key_terms(title_div, doc_key_phrases, 'key_term');
        title_div = mark_key_terms(title_div, search_terms, 'search_term');
        container.append(title_div);
        // Add the abstract
        let abstract_div = $('<div class="col"></div>');
        abstract_div.append($('<span class="fw-bold">Abstract: </span><span>' + doc['Abstract'] + '</span>'));
        abstract_div = mark_key_terms(abstract_div, doc_key_phrases, 'key_term');
        abstract_div = mark_key_terms(abstract_div, search_terms, 'search_term');
        container.append(abstract_div);
        // Add author keywords
        let author_keyword_div = $('<div class="col"></div>');
        let author_keywords = (doc['Author Keywords'] === null) ? "" : doc['Author Keywords'];
        author_keyword_div.append($('<span class="fw-bold">Author Keywords: </span><span>' + author_keywords + '</span>'));
        container.append(author_keyword_div);
        // Add author
        const author_div = $('<div class="col"></div>');
        author_div.append($('<span class="fw-bold">Author: </span><span>' + doc['Authors'] + '</span> '));
        container.append(author_div);
        // Add citation
        const citation_div = $('<div class="col"></div>');
        citation_div.append($('<span><span class="fw-bold">Cited by </span>' + doc['Cited by'] + ' articles</span>'));
        // Add DOI link
        citation_div.append($('<span class="p-3"><span class="fw-bold">DOI </span>' +
            '<a target="_blank" href="https://doi.org/' + doc['DOI'] + '">' + doc['DOI'] + '</a>' +
            '</span>'));
        container.append(citation_div);
    }

    _createUI();
}
