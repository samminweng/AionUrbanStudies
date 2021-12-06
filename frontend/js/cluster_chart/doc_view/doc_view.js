// Create a text view to display the content of the article
function DocView(doc, doc_key_phrases, selected_topics, key_phrases) {
    const searched_topics = (selected_topics === null) ? [] : [selected_topics['topic'], selected_topics['plural']];   // Highlight singular and plural topics
    // console.log(key_phrases);
    const container = $('<div class="card text-dark bg-light"><div class="card-body">' +
        '<p class="card-text"></p></div></div>');
    this.get_container = function () {
        return container;
    }

    // Highlight key terms
    function mark_key_terms(div, terms, class_name) {
        if(terms !== null){
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
        }
        return div;
    }

    function _createUI() {

        // Add Key Phrase
        let key_phrase_div = $('<div><span class="fw-bold">Key Phrases: </span>' + doc_key_phrases.join(", ") + '</div>');
        key_phrase_div = mark_key_terms(key_phrase_div, key_phrases, 'key_term');
        container.find(".card-text").append(key_phrase_div);
        // Add the title
        let title_div = $('<div></div>');
        title_div.append($('<span class="fw-bold">Title: </span><span>' + doc['Title'] + '</span>'));
        // Mark the collocations on title div
        title_div = mark_key_terms(title_div, key_phrases, 'key_term');
        title_div = mark_key_terms(title_div, searched_topics, 'search_term');
        container.find(".card-text").append(title_div);
        // Add the abstract
        let abstract_div = $('<div class="col"></div>');
        const short_abstract = doc['Abstract'].substring(0, 150) + '...';
        abstract_div.append($('<span class="fw-bold">Abstract: </span><span class="abstract">' + short_abstract + '</span>'));
        abstract_div = mark_key_terms(abstract_div, key_phrases, 'key_term');
        abstract_div = mark_key_terms(abstract_div, searched_topics, 'search_term');
        // Add 'more' or 'less' button
        const more_btn = $('<button type="button" class="btn btn-link">more</button>');
        const less_btn = $('<button type="button" class="btn btn-link">less</button>');
        abstract_div.append(more_btn);
        abstract_div.append(less_btn);
        more_btn.show();
        less_btn.hide();
        // Click more btn to display full abstract
        more_btn.on('click', function(event){
            abstract_div.find('.abstract').text(doc['Abstract']);
            abstract_div = mark_key_terms(abstract_div, key_phrases, 'key_term');
            abstract_div = mark_key_terms(abstract_div, searched_topics, 'search_term');
            // Display less btn
            more_btn.hide();
            less_btn.show();
        });
        // Click less btn to display short abstract
        less_btn.on('click', function(event){
            abstract_div.find('.abstract').text(short_abstract);
            abstract_div = mark_key_terms(abstract_div, key_phrases, 'key_term');
            abstract_div = mark_key_terms(abstract_div, searched_topics, 'search_term');
            // Display less btn
            less_btn.hide();
            more_btn.show();
        });
        container.find(".card-text").append(abstract_div);

        // Add citation
        const citation_div = $('<div></div>');
        citation_div.append($('<span><span class="fw-bold">Cited by </span>' + doc['Cited by'] + ' articles</span>'));
        // Add DOI link
        const doi_link = $('<span class="p-3">' +
            '<a target="_blank" href="https://doi.org/' + doc['DOI'] + '">' + doc['DOI'] + '</a>' +
            '</span>');
        citation_div.append(doi_link);
        container.find(".card-text").append(citation_div);
    }

    _createUI();
}

