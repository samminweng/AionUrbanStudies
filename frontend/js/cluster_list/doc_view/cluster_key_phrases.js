// Create a div to display the grouped key phrases
function ClusterKeyPhrase(cluster, cluster_docs, corpus_key_phrases, accordion_div){
    const cluster_key_phrases = cluster['Grouped_Key_Phrases'];
    // Re-order the groups of key phrases
    const outlier_key_phrases = cluster_key_phrases.find(c => c['group'] === -1);
    const grouped_key_phrases = cluster_key_phrases.filter(c => c['group'] !== -1);
    grouped_key_phrases.sort((a, b) => b['count'] - a['count']);
    const all_grouped_key_phrases = grouped_key_phrases.concat([outlier_key_phrases]);
    const total = all_grouped_key_phrases.reduce((pre, cur) => pre + cur['count'], 0);

    // Create an list item to display a group of key phrases
    function createGroupItem(group){
        const group_item = $('<li class="list-group-item d-flex justify-content-between align-items-start"></li>')
        const group_no = group['group'];

        // Display key phrases
        const key_phrases = group['key-phrases'];
        const key_phrase_div = $('<div class="ms-2 me-auto"></div>');
        // Add sub title
        const sub_title_div = $('<div class="fw-bold text-capitalize sub-title"></div>');
        if (group_no === -1){
            sub_title_div.text("miscellaneous");
        }else{
            sub_title_div.text(key_phrases[0]);
        }
        key_phrase_div.append(sub_title_div);
        // Display top 10 key phrases
        const text_span = $('<p class="key_phrase_text"></p>');
        text_span.text(key_phrases.slice(0, 10).join(", "));
        key_phrase_div.append(text_span);
        group_item.append(key_phrase_div);
        // Long list of key phrases
        if(key_phrases.length > 10){
            const btn_div = $('<div class="small"></div>');
            // Create a more btn to view more topics
            const more_btn = $('<span class="text-muted">MORE<span class="ui-icon ui-icon-plus"></span></span>');
            // Create a few btn
            const less_btn = $('<span class="text-muted m-3">LESS<span class="ui-icon ui-icon-minus"></span></span>');
            more_btn.css("font-size", "0.8em");
            less_btn.css("font-size", "0.8em");
            // Display more key phrases
            more_btn.click(function(event){
                const current_key_phrases = text_span.text().split(', ');
                // Display 20 more key phrases
                const max_length = Math.min(key_phrases.length, current_key_phrases.length + 10)
                const more_key_phrases = key_phrases.slice(0, max_length);
                group_item.find('.key_phrase_text').text(more_key_phrases.join(", "));
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
                text_span.text(key_phrases.slice(0, 10).join(", "));
                more_btn.show();
                less_btn.hide();
            });

            // By default, display more btn only.
            more_btn.show();
            less_btn.hide();
            btn_div.append(more_btn);
            btn_div.append(less_btn);
            key_phrase_div.append(btn_div);
        }


        // Add percent
        const percent = Math.round(100 * (group['count']/total));
        const doc_ids = group['DocIds'];
        const group_docs = cluster_docs.filter(d => doc_ids.includes(d['DocId']));
        const percent_btn = $('<button type="button" class="rounded btn-sm">' + percent + '%</button>');
        // Define count btn to display the doc_ids
        percent_btn.click(function(event){
            // Create a doc list
            const doc_list = new DocList(cluster, group_docs, group, corpus_key_phrases);
            document.getElementById('doc_list_heading').scrollIntoView({behavior: "smooth", block: "start", inline: "nearest"});
        });
        group_item.append(percent_btn);
        return group_item;
    }


    // Create a pagination
    // Create a pagination to show the documents
    function createPagination(group_div) {
        // Create the table
        let pagination = $("<div></div>");
        // Add the table header
        const group_list = $('<ul class="list-group list-group-flush"></ul>');
        // // Pagination
        pagination.pagination({
            dataSource: function (done) {
                let result = [];
                for (let i = 0; i < all_grouped_key_phrases.length; i++) {
                    result.push(all_grouped_key_phrases[i]);
                }
                done(result);
            },
            totalNumber: all_grouped_key_phrases.length,
            pageSize: 5,
            showNavigator: true,
            formatNavigator: '<span style="color: #f00"><%= currentPage %></span>/<%= totalPage %> pages, <%= totalNumber %> groups',
            position: 'top',
            className: 'paginationjs-theme-blue paginationjs-small',
            // showGoInput: true,
            // showGoButton: true,
            callback: function (groups, pagination) {
                group_list.empty();
                for (let i = 0; i < groups.length; i++) {
                    const group = groups[i];
                    const group_item = createGroupItem(group);
                    group_list.append(group_item);
                }
            }
        });
        group_div.append(group_list);
        return pagination;
    }



    function _createUI(){
        // Heading
        const heading = $('<h3><span class="fw-bold">Key Phrases</span></h3>');
        const p = $('<p></p>');
        // A list of grouped key phrases
        const group_div = $('<div></div>');
        const pagination = createPagination(group_div);
        p.append(pagination);
        p.append(group_div);
        accordion_div.append(heading);
        accordion_div.append(p);
    }


    _createUI();
}
