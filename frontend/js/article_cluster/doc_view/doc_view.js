// Create a text view to display the content of the article
function DocView(doc, selected_term) {
    const container = $('<div class="card text-dark bg-light">' +
        '<div class="card-body">' +
        '<p class="card-text">' +
        '</p>' +
        '</div></div>');
    this.get_container = function () {
        return container;
    }

    // Highlight key terms
    function mark_key_terms(div, terms, class_name) {
        // function generate_hyph(term)


        // Check if the topic is not empty
        for (const term of terms) {
            if(term !== null){
                // Mark the term
                const mark_options = {
                    "separateWordSearch": false,
                    "accuracy": {
                        "value": "exactly",
                        "limiters": [",", ".", "'s", "/", ";", ":", '(', ')', '‘', '’', '%', 's', 'es', '-']
                    },
                    "acrossElements": true,
                    "ignorePunctuation": ":;.,-–—‒_(){}[]!'\"+=".split(""),
                    "className": class_name
                }
                // div.mark(term, mark_options);
                // Create a regular expression to match hype
                // console.log(term);
                let re_str = term.split(" ").join('[\\s\\-]')
                const reg_exp = new RegExp(re_str);
                div.markRegExp(reg_exp, mark_options);
            }
        }
        return div;
    }

    function _createUI() {
        const doc_key_phrases = doc['KeyPhrases'];
        // Add BERT-based Key Phrase
        const key_phrase_div = $('<p class="container border-info">' +
            '<span class="fw-bold">BERT-driven Keywords: </span>' + doc_key_phrases.join("; ") + '</p>');
        container.find(".card-text").append(key_phrase_div);
        // Add TFIDF terms
        const terms = doc['TFIDFTerms'].map(term => term['term']).slice(0, 5);
        const term_div = $('<p class="container border-info">' +
            '<span class="fw-bold">TFIDF Terms: </span>' + terms.join("; ") + '</p>')
        container.find(".card-text").append(term_div);
        // Add the title
        let title_div = $('<div></div>');
        title_div.append($('<span class="fw-bold">Title: </span><span>' + doc['Title'] + '</span>'));
        // Mark the key phrases on title div
        // title_div = mark_key_terms(title_div, doc_key_phrases, 'key_phrase');
        // title_div = mark_key_terms(title_div, [selected_term], 'search_terms');
        container.find(".card-text").append(title_div);
        // Add the abstract
        let abstract_div = $('<div class="col"></div>');
        // const short_abstract = doc['Abstract'].substring(0, 150) + '...';
        abstract_div.append($('<span class="fw-bold">Abstract: </span><span class="abstract">' + doc['Abstract'] + '</span>'));
        abstract_div = mark_key_terms(abstract_div, doc_key_phrases, 'key_phrase');
        abstract_div = mark_key_terms(abstract_div, [selected_term], 'search_terms');

        container.find(".card-text").append(abstract_div);

        // Add author keywords
        let author_keyword_div = $('<div class="col"></div>');
        author_keyword_div.append($('<span class="fw-bold">Author Keywords: </span><span>' + doc['Author Keywords'] + '</span>'));
        container.find(".card-text").append(author_keyword_div);
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



// // Add 'more' or 'less' button
// const more_btn = $('<button type="button" class="btn btn-link">more</button>');
// const less_btn = $('<button type="button" class="btn btn-link">less</button>');
// abstract_div.append(more_btn);
// abstract_div.append(less_btn);
// more_btn.show();
// less_btn.hide();
// // Click more btn to display full abstract
// more_btn.on('click', function (event) {
//     abstract_div.find('.abstract').text(doc['Abstract']);
//     abstract_div = mark_key_terms(abstract_div, doc_key_phrases, 'key_phrase');
//     abstract_div = mark_key_terms(abstract_div, [selected_term], 'search_terms');
//     // Display less btn
//     more_btn.hide();
//     less_btn.show();
// });
// // Click less btn to display short abstract
// less_btn.on('click', function (event) {
//     abstract_div.find('.abstract').text(short_abstract);
//     abstract_div = mark_key_terms(abstract_div, doc_key_phrases, 'key_phrase');
//     abstract_div = mark_key_terms(abstract_div, [selected_term], 'search_terms');
//     // Display less btn
//     less_btn.hide();
//     more_btn.show();
// });
