// Create a title view
function DocView(doc, highlight_item) {
    const doc_key_phrases = doc['KeyPhrases'];
    let key_phrases = [];
    let search_terms = [];
    if(highlight_item != null){
        if('topic' in highlight_item){
            search_terms = [highlight_item['topic'], highlight_item['plural']]; // Highlight singular and plural topics
        }else if ('group' in highlight_item){
            search_terms = highlight_item['key-phrases'];
        }
        // Filter search terms from doc key phrase
        key_phrases = doc_key_phrases.filter(k => !search_terms.includes(k));
        console.log(highlight_item);
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
                    "limiters": [",", ".", "'s", "/", ";", ":", '(', ')', '‘', '’', '%', 's', 'es']
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
            '<p class="card-text"><span class="lead">' + doc_key_phrases.join(", ") + '</span></p>' +
            '</div></div>');
        container.append(key_phrase_div);

        // Add the title
        let title_div = $('<div></div>');
        title_div.append($('<span class="fw-bold">Title: </span><span>' + doc['Title'] + '</span>'));
        // Mark the collocations on title div
        title_div = mark_key_terms(title_div, key_phrases, 'key_phrase');
        title_div = mark_key_terms(title_div, search_terms, 'search_terms');
        container.append(title_div);
        // Add the abstract
        let abstract_div = $('<div class="col"></div>');
        abstract_div.append($('<span class="fw-bold">Abstract: </span><span>' + doc['Abstract'] + '</span>'));
        abstract_div = mark_key_terms(abstract_div, key_phrases, 'key_phrase');
        abstract_div = mark_key_terms(abstract_div, search_terms, 'search_terms');
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
