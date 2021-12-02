// Create a div to display the grouped key phrases
function ClusterKeyPhrase(cluster_key_phrases){
    let container = $('<div></div>');
    function _createUI(){

        const heading = $('<h5><span class="fw-bold">Key Phrases: </span></h5>');
        const p = $('<p></p>');
        const list = $('<ol class="list-group list-group-flush list-group-numbered"></ol>');
        // Re-order the groups of key phrases
        const grouped_key_phrases = cluster_key_phrases.sort((a, b) => b['group'] - a['group']);
        const total = grouped_key_phrases.reduce((pre, cur) => pre + cur['count'], 0);
        for(const group of grouped_key_phrases){
            const key_phrases = group['key-phrase'].split(", ");
            const top_key_phrases = key_phrases.slice(0, 10);
            const item = $('<li class="list-group-item d-flex justify-content-between align-items-start"></li>');
            const item_div = $('<div class="ms-2 me-auto">' +
                '<div class="key_phrases">' +
                '<span class="key_phrase_text">' + top_key_phrases.join(', ')  +'</span>' +
                '</div></div>');
            // Create a more btn to view more topics
            const more_btn = $('<button type="button" class="btn btn-link">more</button>');
            // Create a few btn
            const less_btn = $('<button type="button" class="btn btn-link">less</button>');
            more_btn.button();
            less_btn.button();
            // Display more key phrases
            more_btn.click(function(event){
                const current_key_phrases = item_div.find('.key_phrase_text').text().split(', ');
                // Display 20 more key phrases
                const max_length = Math.min(key_phrases.length, current_key_phrases.length + 20)
                const more_key_phrases = key_phrases.slice(0, max_length);
                item_div.find('.key_phrase_text').text(more_key_phrases.join(", "));
                if(more_key_phrases.length >= key_phrases.length){
                    // Display 'less' btn only
                    more_btn.hide();
                    less_btn.show();
                }else{
                    more_btn.show();
                    less_btn.show();
                }
            });
            // Display top five key phrases
            less_btn.click(function(event){
                item_div.find('.key_phrase_text').text(top_key_phrases.join(", "));
                more_btn.show();
                less_btn.hide();
            });

            // By default, display more btn only.
            more_btn.show();
            less_btn.hide();
            item_div.find('.key_phrases').append(more_btn);
            item_div.find('.key_phrases').append(less_btn);
            // Add the div to display total number of key phrases
            const count_div = $('<span class="badge bg-primary rounded-pill">' +
                Math.round(100 * (group['count']/total)) + '%</span>');
            item.append(item_div);
            item.append(count_div);
            list.append(item);
        }

        p.append(list);
        container.append(heading);
        container.append(p);
        $('#cluster_key_phrases').empty();
        $('#cluster_key_phrases').append(container);
    }


    _createUI();
}
