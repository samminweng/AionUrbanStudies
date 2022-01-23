// Create a div to display the grouped key phrases
function ClusterLDATopics(cluster_lda_topics, accordion_div){

    // Create an list item to display a group of key phrases
    function createLDATopicView(lda_topic){
        const index = lda_topic['index'];
        const topic_view = $('<li class="list-group-item d-flex justify-content-between align-items-start"></li>')
        // Display key phrases
        const topic_words = lda_topic['lda-topics']['topic'];
        const topic_word_div = $('<div class="ms-2 me-auto"> <div class="fw-bold">Topic #' + (index + 1) +'</div> </div>');
        // Display top 10 key phrases
        const topic_word_span = $('<p class="lda_topic_text"></p>');
        topic_word_span.text(topic_words.join(", "));
        topic_word_div.append(topic_word_span);
        topic_view.append(topic_word_div);
        // // Add percent
        // const percent = Math.round(100 * (group['count']/total));
        // const doc_ids = group['DocIds'];
        // const group_docs = cluster_docs.filter(d => doc_ids.includes(d['DocId']));
        // const percent_btn = $('<button type="button" class="rounded btn-sm">' + percent + '%</button>');
        // // Define count btn to display the doc_ids
        // percent_btn.click(function(event){
        //     // Create a doc list
        //     const doc_list = new DocList(group_docs, group);
        //     document.getElementById('doc_list').scrollIntoView({behavior: "smooth", block: "start", inline: "nearest"});
        // });
        // group_item.append(percent_btn);
        return topic_view;
    }


    // Create a pagination to show the LDA topics by pages
    function createPagination(topic_div) {
        // Create the table
        let pagination = $("<div></div>");
        // Add the table header
        const topic_view_list = $('<ul class="list-group list-group-flush"></ul>');
        // // Pagination
        pagination.pagination({
            dataSource: function (done) {
                let result = [];
                for (let i = 0; i < cluster_lda_topics.length; i++) {
                    result.push({"index": i, "lda-topics": cluster_lda_topics[i]});
                }
                done(result);
            },
            totalNumber: cluster_lda_topics.length,
            pageSize: 10,
            showNavigator: true,
            formatNavigator: '<span style="color: #f00"><%= currentPage %></span>/<%= totalPage %> pages, ' +
                '<%= totalNumber %> LDA Topics',
            position: 'bibe',
            className: 'paginationjs-theme-blue paginationjs-small',
            // showGoInput: true,
            // showGoButton: true,
            callback: function (lda_topics, pagination) {
                topic_view_list.empty();
                for (let i = 0; i < lda_topics.length; i++) {
                    const lda_topic = lda_topics[i];
                    console.log(lda_topic);
                    const lda_topic_views = createLDATopicView(lda_topic);
                    topic_view_list.append(lda_topic_views);
                }
            }
        });
        topic_div.append(topic_view_list);
        return pagination;
    }


    function _createUI(){
        // Heading
        const heading = $('<h3><span class="fw-bold">LDA Topic Modelling</span></h3>');
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
