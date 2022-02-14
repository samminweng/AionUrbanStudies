// Create a div to display a sub-group of key phrases
function KeyPhraseView(sub_group, cluster_docs, color){
    const sub_group_doc_ids = sub_group['DocIds'];
    const key_phrases = sub_group['Key-phrases'];
    const title_words = sub_group['TitleWords'];
    // let title_words = collect_title_words(key_phrases);
    // Get sub_group docs
    const sub_group_docs = cluster_docs.filter(d => sub_group_doc_ids.includes(d['DocId']))
    // console.log(sub_group_docs);
    const word_key_phrase_dict = create_word_key_phrases_dict(title_words, key_phrases);
    const word_doc_dict = create_word_doc_dict(title_words, sub_group_docs);
    // console.log(title_words);

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
        // Sort the title words by the number of
        title_words.sort((a, b) => {
            return dict[b].length - dict[a].length;
        })


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
        dict['Others'] = [];
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
                dict['Others'].push(key_phrase);
            }
        }
        return dict;
    }

    // Collect top 5 frequent from key phrases
    function collect_title_words(key_phrases){
        let word_freq = []
        for(const key_phrase of key_phrases){
            const words = key_phrase.split(" ");
            for (const word of words){
                const found_word = word_freq.find(w => w['word'].toLowerCase() === word.toLowerCase())
                if(found_word){
                    found_word['freq'] += 1;
                }else{
                    if(word === word.toUpperCase()){
                        word_freq.push({'word': word, 'freq':1});
                    }else{
                        word_freq.push({'word': word.toLowerCase(), 'freq':1});
                    }
                }
            }
        }
        // Sort by freq
        word_freq.sort((a, b) => {
            return b['freq'] - a['freq'];
        });
        // Return the top 3
        return word_freq.slice(0, 3).map(a => a['word']);
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
        const list_group = $('<div class="mx-auto"></div>');
        const relevant_key_phrases = word_key_phrase_dict[word];
        // Add each key phrase
        for(const key_phrase of relevant_key_phrases){
            const kp_btn = $('<button type="button" class="btn">' +
                '<span class="badge rounded-pill bg-light text-dark">' + key_phrase+ '</span></button>');
            // Add button
            kp_btn.button({
                classes: {
                    "ui-button": "highlight"
                }
            });
            // On click event
            kp_btn.click(function(event){
                display_papers_by_key_phrase(key_phrase);
            });
            list_group.append(kp_btn);
        }
        key_phrase_div.append(list_group);
        // Display the list of papers containing the key phrases
        display_papers_by_word(word);
    }

    // Create title word header
    function create_header(key_phrase_div){
        // Create a header
        const header_div = $('<div></div>');
        const word_list = title_words.concat(['Others']);
        for(const word of word_list){
            const btn = $('<button type="button" class="btn btn-link">' +
                '<span class="fw-bold text-capitalize" style="color: ' + color+ '">' + word + '</span>' + ' </button>');
            btn.button();
            // click event for sub_group_btn
            btn.click(function(event){
                display_key_phrases_by_word(word, key_phrase_div);
            });
            header_div.append(btn);
        }

        return header_div;
    }

    // Create UI
    function _createUI(){
        // Create an list item to display a group of key phrases
        const container = $('<div></div>');
        // Create a container
        const key_phrase_div = $('<div></div>');
        const header_div = create_header(key_phrase_div);
        // A list of grouped key phrases
        container.append(header_div);
        container.append(key_phrase_div);
        $('#sub_group').empty();
        $('#sub_group').append(container);
        $('#doc_list').empty();
        // Display the docs containing the 1st word
        const word = title_words[0];
        display_key_phrases_by_word(word, key_phrase_div);
    }


    _createUI();
}
