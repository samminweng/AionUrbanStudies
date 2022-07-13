// Create a text view to display the content of the article
function DocView(doc, keywords, color_no) {
    const class_name = 'keyword_' + color_no;
    const container = $('<div class="card text-dark bg-light small m-0 p-0">' +
        '<div class="card-body small">' +
        '<p class="card-text">' +
        '</p>' +
        '</div></div>');
    this.get_container = function () {
        return container;
    }

    // Highlight key terms
    function mark_key_terms(div, terms, class_name) {
        // Check if the topic is not empty
        for (const term of terms) {
            if(term !== null){
                // Mark the term
                const mark_options = {
                    "separateWordSearch": false,
                    "accuracy": {
                        "value": "partial",
                        "limiters": [",", ".", "'s", "/", ";", ":", '(', ')', '‘', '’', '%', 's', 'es', '-']
                    },
                    "acrossElements": true,
                    "ignorePunctuation": ":;.,-–—‒_(){}[]!'\"+=".split(""),
                    "className": class_name,
                }
                const re_str = term.split(" ").join('s?[\\s\\-]');
                // console.log(re_str);
                const reg_exp = new RegExp(re_str, 'i');
                div.markRegExp(reg_exp, mark_options);
            }
        }
        return div;
    }

    function _createUI() {
        const doc_key_phrases = doc['KeyPhrases'];
        // Add BERT-based Key Phrase
        const key_phrase_div = $('<div class="border-info small">' +
            '<span class="fw-bold">Auto-generated Keywords: </span>' + doc_key_phrases.join("; ") + '</div>');
        container.find(".card-text").append(key_phrase_div);
        // // Add TFIDF terms
        // const terms = doc['TFIDFTerms'].map(term => term['term']).slice(0, 5);
        // const term_div = $('<p class="container border-info">' +
        //     '<span class="fw-bold">TFIDF Terms: </span>' + terms.join("; ") + '</p>')
        // container.find(".card-text").append(term_div);
        // Add the title
        let title_div = $('<div></div>');
        title_div.append($('<span class="fw-bold">Title: </span><span class="small">' + doc['Title'] + '</span>'));

        container.find(".card-text").append(title_div);
        // Add the abstract
        let abstract_div = $('<div class="col"></div>');
        // const short_abstract = doc['Abstract'].substring(0, 150) + '...';
        abstract_div.append($('<span class="fw-bold">Abstract: </span><span class="abstract small">' + doc['Abstract'] + '</span>'));
        abstract_div = mark_key_terms(abstract_div, doc_key_phrases, 'key_phrase');
        abstract_div = mark_key_terms(abstract_div, keywords, class_name);

        container.find(".card-text").append(abstract_div);

        // Add author keywords
        // let author_keyword_div = $('<div class="col"></div>');
        // author_keyword_div.append($('<span class="fw-bold">Author Keywords: </span><span>' + doc['Author Keywords'] + '</span>'));
        // container.find(".card-text").append(author_keyword_div);
        // Add authors
        let author_div = $('<div class="col"></div>');
        author_div.append($('<span class="fw-bold">Authors: </span><span>' + doc['Authors'] + '</span>'));
        container.find(".card-text").append(author_div);

        // Add citation
        const paper_info_div = $('<div></div>');
        paper_info_div.append($('<span><span class="fw-bold">Cited by </span>' + doc['Cited by'] + ' articles</span>'));
        // Add Year
        paper_info_div.append($('<span><span class="fw-bold"> Year </span>' + doc['Year'] + ' </span>'))
        // Add DOI link
        paper_info_div.append($('<span><span class="fw-bold"> DOI </span>' +
            '<a target="_blank" href="https://doi.org/' + doc['DOI'] + '">' + doc['DOI'] + '</a>' +
            '</span>'));
        container.find(".card-text").append(paper_info_div);
    }

    _createUI();
}

