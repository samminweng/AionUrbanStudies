// Create a div to display the grouped key phrases
function KeyPhraseListView(group, sub_groups, total, cluster_docs){
    console.log(group);
    console.log(sub_groups);
    // console.log(cluster_key_phrases);
    // const total = group['NumPhrases'];

    // Highlight key terms
    function mark_key_terms(div, terms, class_name) {
        if(terms !== null){
            // Check if the topic is not empty
            for (const term of terms) {
                // Mark the topic
                const mark_options = {
                    "separateWordSearch": false,
                    "accuracy": {
                        "value": "exactly",
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

    // Display the relevant paper
    function display_relevant_papers(sub_group){
        const sub_group_doc_ids = sub_group['DocIds'];
        const key_phrases =  sub_group['Key-phrases'];
        // Get sub_group docs
        const sub_group_docs = cluster_docs.filter(d => sub_group_doc_ids.includes(d['DocId']))
        console.log(sub_group_docs);
        const header_text = 'about ' + sub_group['TitleWords'].join(", ");
        // Create a list view of docs
        const doc_list = new DocList(sub_group_docs, key_phrases, header_text);
    }


    // Create an list item to display a group of key phrases
    function createSubGroup(sub_group){
        const sub_group_item = $('<li class="list-group-item d-flex justify-content-between align-items-start"></li>')
        // Display key phrases
        const key_phrases = sub_group['Key-phrases'];
        const title_terms = sub_group['TitleWords'];
        const sub_group_docs = sub_group['DocIds'];
        const sub_group_btn = $('<a class="ui-widget ui-corner-all fw-bold text-capitalize ms-0 mb-3">' +
                                ' ' + title_terms.join(", ") + ' (' + key_phrases.length + ')</a>');
        sub_group_btn.button();
        // click event for sub_group_btn
        sub_group_btn.click(function(event){
            // Display all the relevant papers.
            display_relevant_papers(sub_group);
        });

        const key_phrase_div = $('<div class="ms-2 me-auto"></div>');
        key_phrase_div.append(sub_group_btn);
        // Display top 10 key phrases
        const text_span = $('<p class="key_phrase_text"></p>');
        const MAX = 10
        text_span.text(key_phrases.slice(0, MAX).join(", "));
        key_phrase_div.append(text_span);

        sub_group_item.append(key_phrase_div);
        // // Long list of key phrases
        if(key_phrases.length > MAX){
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
                text_span.text(more_key_phrases.join(", "));
                mark_key_terms(text_span, title_terms, 'key_phrase');
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
                text_span.text(key_phrases.slice(0, MAX).join(", "));
                mark_key_terms(text_span, title_terms, 'key_phrase');
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
        // Highlight the title words in each individual sub-group
        mark_key_terms(text_span, title_terms, 'key_phrase');
        return sub_group_item;
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
                for (let i = 0; i < sub_groups.length; i++) {
                    result.push(sub_groups[i]);
                }
                done(result);
            },
            totalNumber: sub_groups.length,
            pageSize: 10,
            showNavigator: true,
            formatNavigator: '<span style="color: #f00"><%= currentPage %></span>/<%= totalPage %> pages, ' +
                '<%= totalNumber %> sub-groups of key phrases',
            position: 'top',
            className: 'paginationjs-theme-black paginationjs-small',
            // showGoInput: true,
            // showGoButton: true,
            callback: function (sub_groups, pagination) {
                group_list.empty();
                for (let i = 0; i < sub_groups.length; i++) {
                    const sub_group = sub_groups[i];
                    const sub_group_view = createSubGroup(sub_group);
                    group_list.append(sub_group_view);
                }
            }
        });
        group_div.append(group_list);
        return pagination;
    }

    // Create header
    function createHeader(){
        const group_id = group['Group'];
        const group_docs = group['DocIds'];
        const group_key_phrases = group['Key-phrases'];
        $('#sub_group_header').empty();
        const header = $('<div class="bg-secondary bg-opacity-10 p-3"></div>')
        // Create a header
        header.append($('<div class="row">' +
                            '<div class="col">Group #' + (group_id + 1) + ' has ' + group_key_phrases.length + ' phrases</div>' +
                        '</div>'));
        $('#sub_group_header').append(header);

    }

    // Create UI
    function _createUI(){
        $('#sub_group_list_view').empty();
        const p = $('<p></p>');
        // A list of grouped key phrases
        const group_div = $('<div></div>');
        const pagination = createPagination(group_div);
        p.append(pagination);
        p.append(group_div);
        $('#sub_group_list_view').append(p);
        createHeader();
    }


    _createUI();
}
