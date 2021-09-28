// Create a title view
function TextView(doc, topics) {

    const container = $('<div class="small"></div>');
    this.get_container = function () {
        return container;
    }

    // Highlight key terms
    function mark_key_terms(div, terms, class_name) {
        // Check if the topic is not empty
        for(const term of terms){
            // Mark the topic
            const mark_options = {
                "separateWordSearch": false,
                "accuracy": {
                    "value": "exactly",
                    "limiters": [",", ".", "'s", "/", ";", ":"]
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
        // Key terms
        let key_terms = doc['HDBSCAN_Cluster_KeyTerms'].slice(0, 5);
        let key_term_div = $('<div><h3><span class="fw-bold">Top 5 key terms: </span>' + key_terms.join("; ") + '</h3>' +
            '<div><p>' + doc['HDBSCAN_Cluster_KeyTerms'].join("; ")+ '</p></div>' +
            '</div>');
        key_term_div.accordion({
            icons: null,
            collapsible: true,
            heightStyle: "content"
        });
        container.append(key_term_div);
        if(topics.length>0){
            key_terms = key_terms.filter(term => term !== topics[0]);// Remove the topic term
        }
        // Add the title
        let title_div = $('<div></div>');
        title_div.append($('<span class="fw-bold">Title: </span><span>' + doc['Title'] + '</span>'));
        // Mark the collocations on title div
        title_div = mark_key_terms(title_div, topics, 'search_term');
        title_div = mark_key_terms(title_div, key_terms, 'key_term');
        container.append(title_div);
        // Add the abstract
        let abstract_div = $('<div class="col"></div>');
        abstract_div.append($('<span class="fw-bold">Abstract: </span><span>' + doc['Abstract'] + '</span>'));
        abstract_div = mark_key_terms(abstract_div, topics, 'search_term');
        abstract_div = mark_key_terms(abstract_div, key_terms, 'key_term');
        container.append(abstract_div);
        // Add the author keyowrds
        let author_keyword_div = $('<div class="col"></div>');
        let author_keywords = (doc['Author Keywords'] === null)? "": doc['Author Keywords'];
        author_keyword_div.append($('<span class="fw-bold">Author Keywords: </span><span>' + author_keywords + '</span>'));
        container.append(author_keyword_div);
        // author_keyword_div = mark_key_terms(author_keyword_div, [searched_term], 'search_term');
        // author_keyword_div = mark_key_terms(author_keyword_div, complementary_terms, 'complementary_term');

    }

    _createUI();
}
