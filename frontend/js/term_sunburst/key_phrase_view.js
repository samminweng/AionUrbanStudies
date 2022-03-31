// Create a div to display a sub-group of key phrases
function KeyPhraseView(group, cluster_docs, select_index){
    const group_doc_ids = group['DocIds'];
    const key_phrases = group['key-phrases'];
    const topic_words = group['topic_words'].concat("others");
    // Get group docs
    const group_docs = cluster_docs.filter(d => group_doc_ids.includes(d['DocId']))
    // console.log(group_docs);
    const word_key_phrase_dict = Utility.create_word_key_phrases_dict(topic_words, key_phrases);
    const word_doc_dict = Utility.create_word_doc_dict(topic_words, group_docs, word_key_phrase_dict);
    // console.log(key_phrases);

    // Highlight key terms
    function mark_key_terms(div, terms, class_name) {
        if(terms !== null){
            // Check if the topic is not empty
            for (const term of terms) {
                // Mark the topic
                const mark_options = {
                    "separateWordSearch": false,
                    "accuracy": {
                        "value": "partially",
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

    // Display the relevant papers containing key phrases
    function display_papers_by_word(word){
        const relevant_doc_ids = word_doc_dict[word];
        const relevant_key_phrases = word_key_phrase_dict[word];
        const relevant_docs = group_docs.filter(d => relevant_doc_ids.includes(d['DocId']));

        // Display all the relevant papers.
        const header_text = word;
        // Create a list view of docs
        const doc_list = new DocList(relevant_docs, relevant_key_phrases, header_text);
    }

    // Display the relevant papers containing key phrases
    function display_papers_by_key_phrase(key_phrase){
        const relevant_docs = group_docs.filter(d => {
            const found = d['KeyPhrases'].find(kp => kp.toLowerCase() === key_phrase.toLowerCase());
            if(found){
                return true;
            }
            return false;
        });

        // Display all the relevant papers.
        const header_text = key_phrase;
        // Create a list view of docs
        const doc_list = new DocList(relevant_docs, [key_phrase], header_text);
    }

    // Display the key phrase of a word
    function display_key_phrases_by_word(word, key_phrase_div){
        key_phrase_div.empty();
        const relevant_key_phrases = word_key_phrase_dict[word];
        // Get the number of rows by rounding upto the closest integer
        const num_row = Math.ceil(relevant_key_phrases.length/3) + 1;
        // Add each key phrase
        for(let i=0; i<num_row; i++){
            // Create a new row
            const row = $('<div class="row"></div>');
            key_phrase_div.append(row);
            let j = 0;
            for(j=i*3; j <i*3 + 3 && j < relevant_key_phrases.length; j++){
                const key_phrase = relevant_key_phrases[j];
                const kp_btn = $('<button type="button" class="btn">' + key_phrase+ '</button>');
                // Add button
                kp_btn.button();
                // On click event
                kp_btn.click(function(event){
                    display_papers_by_key_phrase(key_phrase);
                });
                row.append($('<div class="col border-1 border-start border-dark"></div>').append(kp_btn));
            }
            if(j === relevant_key_phrases.length){
                // Add empty cell if needed
                let reminders = relevant_key_phrases.length%3;
                while(reminders > 0 && reminders !== 3){
                    row.append($('<div class="col border-1 border-start border-dark"></div>'));
                    reminders = reminders + 1;
                }
            }
        }
        // Highlight the word
        key_phrase_div = mark_key_terms(key_phrase_div, [word], 'key_term');

        // Display the list of papers containing the key phrases
        display_papers_by_word(word);
    }

    // Create title word header using navs & tabs
    function create_title_word_header(key_phrase_div){
        // Create a header
        const nav = $('<nav class="nav nav-pills"></nav>');
        const word_list = topic_words;
        for(let i=0; i < word_list.length; i++){
            const word = word_list[i];
            const relevant_key_phrases = word_key_phrase_dict[word];
            if(relevant_key_phrases.length > 0){
                const btn = $('<a class="nav-link">' +
                    '<span class="fw-bold">' + word + '</span></a>');
                if(i === select_index){
                    btn.addClass("active");     // Set default tab
                }
                // click event for sub_group_btn
                btn.click(function(event){
                    header_div.find("a").removeClass("active");
                    display_key_phrases_by_word(word, key_phrase_div);
                    btn.addClass("active");
                });
                nav.append(btn);
            }
        }
        // Add header div
        const header_div = $('<div class="container p-3"></div>');
        header_div.append(nav);
        return header_div;
    }

    // Create UI
    function _createUI(){
        // Create an list item to display a group of key phrases
        const container = $('<div></div>');
        // Create a container
        const key_phrase_div = $('<div class="container"></div>');
        const header_div = create_title_word_header(key_phrase_div);
        // A list of grouped key phrases
        container.append(header_div);
        container.append(key_phrase_div);
        $('#group').empty();
        $('#group').append(container);
        $('#doc_list').empty();
        // Display the docs containing the 1st word
        const word = topic_words[select_index];
        display_key_phrases_by_word(word, key_phrase_div);
    }

    _createUI();
}
