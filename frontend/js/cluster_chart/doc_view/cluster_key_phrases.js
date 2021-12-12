// Create a div to display the grouped key phrases
function ClusterKeyPhrase(cluster_key_phrases, cluster_docs, corpus_key_phrases, accordion_div){
    // Re-order the groups of key phrases
    const outlier_key_phrases = cluster_key_phrases.find(c => c['group'] === -1);
    const grouped_key_phrases = cluster_key_phrases.filter(c => c['group'] !== -1);
    grouped_key_phrases.sort((a, b) => b['count'] - a['count']);
    const all_grouped_key_phrases = grouped_key_phrases.concat([outlier_key_phrases]);
    const total = all_grouped_key_phrases.reduce((pre, cur) => pre + cur['count'], 0);

    // Create a pagination
    // Create a pagination to show the documents
    function createPagination(group_table) {
        // Create the table
        let pagination = $("<div></div>");
        // Add the table header
        const group_list = $('<ul class="list-group list-group-flush"></ul>');
        // Display each grouped key phrases
        for(const group of all_grouped_key_phrases){
            const item = $('<li class="list-group-item d-flex justify-content-between align-items-start"></li>')
            const group_no = group['group'];
            const percent = Math.round(100 * (group['count']/total));
            const top_key_phrases = group['key-phrases'].slice(0, 10);
            const key_phrase_div = $('<div class="ms-2 me-auto"></div>');
            // Add sub title
            const sub_title_div = $('<div class="fw-bold sub-title"></div>');
            if (group_no === -1){
                sub_title_div.text("Un-grouped Key Phrases:");
            }
            key_phrase_div.append(sub_title_div);
            // Display top 10 key phrases
            const text_span = $('<span class="key_phrase_text"></span>');
            text_span.text(top_key_phrases.join(", "));
            key_phrase_div.append(text_span);
            item.append(key_phrase_div);
            // Add percent
            const span = $('<span class="badge bg-primary rounded-pill"></span>')
            span.text(" (" + percent+ '%)')
            item.append(span);
            group_list.append(item);
        }
        group_table.append(group_list);

        //
        //
        //     const top_key_phrases = key_phrases.slice(0, 10);
        //     const item = $('<li class="list-group-item d-flex justify-content-between align-items-start"></li>');
        //     const item_div = $('<div class="ms-2 me-auto">' +
        //         '<div class="key_phrases">' +
        //         '<span class="key_phrase_text">' + top_key_phrases.join(', ')  +'</span>' +
        //         '</div></div>');
        //     // Create a more btn to view more topics
        //     const more_btn = $('<button type="button" class="btn btn-link">more</button>');
        //     // Create a few btn
        //     const less_btn = $('<button type="button" class="btn btn-link">less</button>');
        //     more_btn.button();
        //     less_btn.button();
        //     // Display more key phrases
        //     more_btn.click(function(event){
        //         const current_key_phrases = item_div.find('.key_phrase_text').text().split(', ');
        //         // Display 20 more key phrases
        //         const max_length = Math.min(key_phrases.length, current_key_phrases.length + 20)
        //         const more_key_phrases = key_phrases.slice(0, max_length);
        //         item_div.find('.key_phrase_text').text(more_key_phrases.join(", "));
        //         if(more_key_phrases.length >= key_phrases.length){
        //             // Display 'less' btn only
        //             more_btn.hide();
        //             less_btn.show();
        //         }else{
        //             more_btn.show();
        //             less_btn.show();
        //         }
        //     });
        //     // Display top five key phrases
        //     less_btn.click(function(event){
        //         item_div.find('.key_phrase_text').text(top_key_phrases.join(", "));
        //         more_btn.show();
        //         less_btn.hide();
        //     });
        //
        //     // By default, display more btn only.
        //     more_btn.show();
        //     less_btn.hide();
        //     item_div.find('.key_phrases').append(more_btn);
        //     item_div.find('.key_phrases').append(less_btn);
        //
        //     // Get the doc ids that contain this grouped key phrases
        //     const doc_ids = group['DocIds'];
        //     const group_docs = cluster_docs.filter(d => doc_ids.includes(d['DocId']));
        //     console.log(group_docs);
        //
        //     // Add the div to display total number of key phrases
        //     const count_btn = $('<button class="badge bg-primary rounded-pill">' + percent + '%</button>');
        //     count_btn.button();
        //     // Define count btn to display the doc_ids
        //     count_btn.click(function(event){
        //         // Create a doc list
        //         const doc_list = new DocList(group_docs, null, corpus_key_phrases, key_phrases);
        //         document.getElementById('doc_list').scrollIntoView({behavior: "smooth", block: "start", inline: "nearest"});
        //     });
        //
        //     item.append(item_div);
        //     item.append(count_btn);
        //     list.append(item);
        // }
        return pagination;
    }



    function _createUI(){
        // Heading
        const heading = $('<h3>Key Phrases</h3>');
        const p = $('<p></p>');
        // A list of grouped key phrases
        const group_table = $('<div></div>');
        const pagination = createPagination(group_table);
        p.append(group_table);
        p.append(pagination);
        accordion_div.append(heading);
        accordion_div.append(p);
    }


    _createUI();
}
