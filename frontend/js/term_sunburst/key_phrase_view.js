// Create a div to display a sub-group of key phrases
function KeyPhraseView(sub_group, cluster_docs, color){
    const sub_group_doc_ids = sub_group['DocIds'];
    const key_phrases = sub_group['Key-phrases'];
    const title_words = sub_group['TitleWords'];
    // Get sub_group docs
    const sub_group_docs = cluster_docs.filter(d => sub_group_doc_ids.includes(d['DocId']))
    // console.log(sub_group_docs);
    const word_key_phrase_dict = create_word_key_phrases_dict(title_words, key_phrases);
    const word_doc_dict = create_word_doc_dict(title_words, sub_group_docs);
    // console.log(key_phrases);

    // Create a dict to store the word and doc relation
    function create_word_doc_dict(title_words, sub_group_docs){
        let dict = {};
        for(const [title_word, key_phrases] of Object.entries(word_key_phrase_dict)){
            let word_doc_ids = [];
            // Match the key phrases with title words and assign key phrase
            for(const key_phrase of key_phrases) {
                // // Collect the doc containing the key phrase
                const relevant_docs = sub_group_docs.filter(d => {
                    const found = d['KeyPhrases'].find(kp => kp.toLowerCase() === key_phrase.toLowerCase());
                    if(found){
                        return true;
                    }
                    return false;
                });
                for (const doc of relevant_docs){
                    if(!word_doc_ids.includes(doc['DocId'])){
                        word_doc_ids.push(doc['DocId']);
                    }
                }
            }
            dict[title_word] = word_doc_ids;
        }
        return dict;
    }

    // Create a dict of word and key phrases
    // Allocate the key phrases based on the words
    function create_word_key_phrases_dict(title_words, key_phrases){
        let dict = {};
        // Initialise dict with an array
        for(const title_word of title_words){
            dict[title_word] = [];
        }
        // Add misc
        dict['others'] = [];
        // Match the key phrases with title words and assign key phrase
        for(const key_phrase of key_phrases) {
            let is_found = false;
            for(const title_word of title_words){
                if(key_phrase.toLowerCase().includes(title_word.toLowerCase())){
                    dict[title_word].push(key_phrase);
                    is_found = true;
                }
            }
            // If not found, assign to 'misc'
            if(!is_found){
                dict['others'].push(key_phrase);
            }
        }
        return dict;
    }


    // Display the relevant papers containing key phrases
    function display_papers_by_word(word){
        const relevant_doc_ids = word_doc_dict[word];
        const relevant_key_phrases = word_key_phrase_dict[word];
        const relevant_docs = sub_group_docs.filter(d => relevant_doc_ids.includes(d['DocId']));

        // Display all the relevant papers.
        const header_text = word;
        // Create a list view of docs
        const doc_list = new DocList(relevant_docs, relevant_key_phrases, header_text);
    }

    // Display the relevant papers containing key phrases
    function display_papers_by_key_phrase(key_phrase){
        const relevant_docs = sub_group_docs.filter(d => {
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
        let row = $('<div class="row"></div>');
        key_phrase_div.append(row);
        // Add each key phrase
        for(let i=0; i<relevant_key_phrases.length; i++){
            // Create a new row
            if(i>0 && (i%3) === 0){
                key_phrase_div.append(row);
                row = $('<div class="row"></div>');
            }
            const key_phrase = relevant_key_phrases[i];
            const kp_btn = $('<button type="button" class="btn">' + key_phrase+ '</button>');
            // Add button
            kp_btn.button();
            // On click event
            kp_btn.click(function(event){
                display_papers_by_key_phrase(key_phrase);
            });
            row.append($('<div class="col border-1 border-start border-dark"></div>').append(kp_btn));
        }

        // Add empty cell if needed
        let reminders = relevant_key_phrases.length%3;
        while(reminders > 0 && reminders !== 3){
            row.append($('<div class="col border-1 border-start border-dark"></div>'));
            reminders = reminders + 1;
        }


        // Display the list of papers containing the key phrases
        display_papers_by_word(word);
    }

    // Create title word header using navs & tabs
    function create_title_word_header(key_phrase_div){
        // Create a header
        const nav = $('<nav class="nav nav-pills"></nav>');
        const word_list = title_words.concat(['others']);
        for(let i=0; i < word_list.length; i++){
            const word = word_list[i];
            const relevant_key_phrases = word_key_phrase_dict[word];
            if(relevant_key_phrases.length > 0){
                const btn = $('<a class="nav-link">' +
                    '<span class="fw-bold">' + word + '</span></a>');
                if(i === 0){
                    btn.addClass("active");     // Set default tab
                }
                // '<span class="fw-bold" style="color: ' + color+ '">' + word + '</span></a>');
                // btn.button();
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
        $('#sub_group').empty();
        $('#sub_group').append(container);
        $('#doc_list').empty();
        // Display the docs containing the 1st word
        const word = title_words[0];
        display_key_phrases_by_word(word, key_phrase_div);
        const element = document.getElementById("sub_group");
        element.scrollIntoView({behavior: "smooth", block: "end", inline: "nearest"});



    }


    _createUI();
}
