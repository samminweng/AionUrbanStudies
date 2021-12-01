// Create a div to display the grouped key phrases
function ClusterKeyPhrase(cluster_key_phrases, accordion_div){

    function _createUI(){
        const heading = $('<h3><span class="fw-bold">Key Phrases: </span></h3>');
        const p = $('<p></p>');
        const list = $('<ol class="list-group list-group-flush list-group-numbered"></ol>');
        for(const group of cluster_key_phrases){
            const key_phrases = group['key-phrase'].split(", ");
            const top_key_phrases = key_phrases.slice(0, 5);
            const item = $('<li class="list-group-item d-flex justify-content-between align-items-start"></li>');
            const item_div = $('<div class="ms-2 me-auto">' +
                '<div class="key_phrases">' +
                '<div class="key_phrase_text">' + top_key_phrases.join(', ')  +'</div>' +
                '</div></div>');
            // Create a more btn to view more topics
            const more_btn = $('<button><span class="ui-icon ui-icon-plus"></button>');
            more_btn.button();
            more_btn.click(function(event){
                // Display 30 more key phrases
                const max_length = Math.min(key_phrases.length, 30)
                const more_key_phrases = key_phrases.slice(0, max_length);
                item_div.find('.key_phrase_text').text(more_key_phrases.join(", "));
                more_btn.hide();
                few_btn.show();
            });
            // Create a few btn
            const few_btn = $('<button><span class="ui-icon ui-icon-minus"></button>');
            few_btn.button();
            few_btn.click(function(event){
                item_div.find('.key_phrase_text').text(top_key_phrases.join(", "));
                more_btn.show();
                few_btn.hide();
            });


            more_btn.show();
            few_btn.hide();
            item_div.find('.key_phrases').append(more_btn);
            item_div.find('.key_phrases').append(few_btn);
            // Add the div to display total number of key phrases
            const count_div = $('<span class="badge bg-primary rounded-pill">' + group['count']+ '</span>');
            item.append(item_div);
            item.append(count_div);
            list.append(item);
        }

        p.append(list);
        accordion_div.append(heading);
        accordion_div.append($('<div></div>').append(p));
    }


    _createUI();
}
