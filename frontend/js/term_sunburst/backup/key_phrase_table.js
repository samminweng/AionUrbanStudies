// Create a div table to display a sub-group of key phrases
function KeyPhraseTable(sub_group, cluster_docs){
    const sub_group_doc_ids = sub_group['DocIds'];
    const key_phrases = sub_group['Key-phrases'];
    const title_words = sub_group['TitleWords'];
    console.log(key_phrases);
    // Get sub_group docs
    const sub_group_docs = cluster_docs.filter(d => sub_group_doc_ids.includes(d['DocId']))
    console.log(sub_group_docs);

    // Allocate the key phrases based on the words
    function create_key_phrases_dict(){
        let key_phrase_dict = {};
        // Initialise dict with an array
        for(const title_word of title_words){
            key_phrase_dict[title_word] = [];
        }
        // Add misc
        key_phrase_dict['misc'] = [];
        // Match the key phrases with title words and assign key phrase
        let max_count = 0;
        for(const key_phrase of key_phrases) {
            const arr = key_phrase.split(" ");
            let is_found = false;
            for (let i = arr.length - 1; i >= 0; i--) {
                const word = arr[i];
                const found = title_words.find(title_word => title_word.toLowerCase() === word.toLowerCase());
                if (found) {
                    key_phrase_dict[found].push(key_phrase);
                    is_found = true;
                    break;
                }
            }
            // If not found, assign to 'misc'
            if(!is_found){
                key_phrase_dict['misc'].push(key_phrase);
            }
        }
        // Get maximal count
        for(const [title_word, list] of Object.entries(key_phrase_dict)){
            max_count = Math.max(list.length, max_count);
        }
        return [key_phrase_dict, max_count];
    }

    // Create title word header
    function create_header(){
        // Create a header
        const header_div = $('<div></div>');
        const btn = $('<a class="ui-widget ui-corner-all fw-bold text-capitalize">' +
                    '<span>' + title_words.join(", ")+ '</span> (' + sub_group_docs.length +' papers) </a>');
        btn.button();
        // click event for sub_group_btn
        btn.click(function(event){
            // Display all the relevant papers.
            const header_text = title_words.join(", ");
            // Create a list view of docs
            const doc_list = new DocList(sub_group_docs, key_phrases, header_text);
        });
        header_div.append(btn);
        return header_div;
    }

    // Create table header
    function create_table_header(){
        // Create table header with Title words
        const table_header_div = $('<thead><tr></tr></thead>');
        for(const word of title_words){
            table_header_div.find('tr').append($('<th scope="col" class="text-capitalize">' + word + ' </th>'));
        }
        // Add misc
        table_header_div.find('tr').append($('<th scope="col" class="text-capitalize">misc</th>'));
        return table_header_div;
    }

    // Display papers for each individual key phrase
    function display_paper_by_key_phrase(key_phrase){
        const docs = sub_group_docs.filter(d => {
            const found = d['KeyPhrases'].find(kp => kp.toLowerCase() === key_phrase.toLowerCase())
            return found;
        });
        console.log(docs);
        // Create a list view of docs
        const doc_list = new DocList(docs, [key_phrase], key_phrase);
    }

    // Create an list item to display a group of key phrases
    function createSubGroupTable(){
        // Create a container
        const container = $('<div></div>');
        const header_div = create_header();
        container.append(header_div);
        // Create a table
        const table_div = $('<table class="table table-bordered table-sm">' +
                            '<tbody></tbody></table>');
        const table_header_div = create_table_header();
        // Append div
        table_div.append(table_header_div);
        // Create table body
        // Get all the key-phrases
        const [key_phrase_dict, max_count] = create_key_phrases_dict(key_phrases, title_words);
        console.log(key_phrase_dict);
        // Add the key-phrases
        for(let i = 0; i < max_count; i++){
            // Get the
            const row = $('<tr></tr>');
            for(const word of title_words){
                const key_phrase = key_phrase_dict[word][i];
                const col = $('<td></td>');
                if(key_phrase){
                    const col_btn = $('<a class="ui-widget ui-corner-all">'  + key_phrase+ ' </a>');
                    col_btn.button();
                    col_btn.click(function(event){
                        display_paper_by_key_phrase(key_phrase);
                    });
                    // col.text(key_phrase);
                    col.append(col_btn);
                }
                row.append(col);
            }
            // Add misc
            const key_phrase = key_phrase_dict['misc'][i];
            const col = $('<td></td>');
            if(key_phrase){
                const col_btn = $('<a class="ui-widget ui-corner-all">'  + key_phrase+ ' </a>');
                col_btn.button();
                col_btn.click(function(event){
                    display_paper_by_key_phrase(key_phrase);
                });
                col.append(col_btn);
            }
            row.append(col);
            table_div.append(row);
        }
        container.append(table_div);
        return container;
    }

    // Create UI
    function _createUI(){
        const p = $('<p></p>');
        // A list of grouped key phrases
        const table_div = createSubGroupTable();
        p.append(table_div);
        $('#sub_group').empty();
        $('#sub_group').append(p);
        $('#doc_list').empty();
        // Create a list of docs
        const doc_list = new DocList(sub_group_docs, key_phrases, title_words.join(", "));
        document.getElementById('sub_group').scrollIntoView({behavior: "smooth", block: "start", inline: "nearest"});
    }


    _createUI();
}
