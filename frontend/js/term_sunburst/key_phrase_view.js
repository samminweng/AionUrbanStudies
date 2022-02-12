// Create a div to display a sub-group of key phrases
function KeyPhraseView(sub_group, cluster_docs, color){
    const sub_group_doc_ids = sub_group['DocIds'];
    const key_phrases = sub_group['Key-phrases'];
    // const title_words = sub_group['TitleWords'];
    const title_words = collect_title_words(key_phrases);
    console.log(title_words);
    // Get sub_group docs
    const sub_group_docs = cluster_docs.filter(d => sub_group_doc_ids.includes(d['DocId']))
    console.log(sub_group_docs);
    const dict = create_key_phrases_dict(title_words, key_phrases);

    // Create a dict of word and key phrases
    // Allocate the key phrases based on the words
    function create_key_phrases_dict(title_words, key_phrases){
        let dict = {};
        // Initialise dict with an array
        for(const title_word of title_words){
            dict[title_word] = [];
        }
        // Add misc
        dict['Others'] = [];
        // Match the key phrases with title words and assign key phrase
        let max_count = 0;
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
            // if(a['freq'] === b['freq']){
            //     return a['word'].localeCompare(b['word']);
            // }
            return b['freq'] - a['freq'];
        });
        // Return the top 5
        return word_freq.slice(0, 5).map(a => a['word']);
    }


    // Display the relevant papers containing key phrases
    function display_papers_by_word(word){
        const relevant_key_phrases = dict[word];
        const relevant_docs = sub_group_docs.filter(d => {
            for(const key_phrase of relevant_key_phrases){
                const found = d['KeyPhrases'].find(kp => kp.toLowerCase() === key_phrase.toLowerCase());
                if(found){
                    return true;
                }
            }
            return false;
        })

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
        })

        // Display all the relevant papers.
        const header_text = key_phrase;
        // Create a list view of docs
        const doc_list = new DocList(relevant_docs, [key_phrase], header_text);
    }

    // Create title word header
    function create_header(key_phrase_div){
        // Create a header
        const header_div = $('<div></div>');
        const word_list = title_words.concat(['Others']);
        for(const word of word_list){
            const relevant_key_phrases = dict[word];
            const btn = $('<button type="button" class="btn btn-lg">' +
                '<span class="fw-bold text-capitalize" style="color: ' + color+ '">' + word + '</span> </button>');
            btn.button();
            // click event for sub_group_btn
            btn.click(function(event){
                // Display a list of relevant key phrases
                key_phrase_div.empty();
                const list_group = $('<div class="mx-auto"></div>');
                // Add each key phrase
                for(const key_phrase of relevant_key_phrases){
                    const kp_btn = $('<button type="button" class="btn btn-lg">' +
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

            });
            header_div.append(btn);
        }

        return header_div;
    }


    // Create an list item to display a group of key phrases
    function createSubGroupView(){
        // Create a container
        const container = $('<div></div>');
        const key_phrase_div = $('<div></div>');

        // Add the header div to display the title words
        const header_div = create_header(key_phrase_div);
        container.append(header_div);
        container.append(key_phrase_div);
        return container;
    }

    // Create UI
    function _createUI(){
        const p = $('<p></p>');
        // A list of grouped key phrases
        const div = createSubGroupView();
        p.append(div);
        $('#sub_group').empty();
        $('#sub_group').append(p);
        $('#doc_list').empty();
        // Create a list of docs
        const doc_list = new DocList(sub_group_docs, key_phrases, title_words.join(", "));
        document.getElementById('sub_group').scrollIntoView({behavior: "smooth", block: "start", inline: "nearest"});
    }


    _createUI();
}
