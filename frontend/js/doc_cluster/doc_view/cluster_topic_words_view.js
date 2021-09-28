// Display a list of topic words for a cluster
function ClusterTopicWordsView(cluster){
    const cluster_no = cluster['Cluster'];
    const cluster_docs = Utility.get_documents_by_doc_ids(cluster['DocIds']);
    // Create a pagination
    function createPagination(docTable, topics_tf_idf, topics_collocation, topics_topic2vec) {
        // Create the table
        let pagination = $("<div></div>");
        // Pagination
        pagination.pagination({
            dataSource: function (done) {
                let results = [];
                for (let i = 0; i < topics_tf_idf.length; i++) {
                    const tf_idf_term = topics_tf_idf[i]['topic_words'];
                    const collocation = (i < topics_collocation.length)? topics_collocation[i]['topic_words']: "";
                    const topic2vec = (i < topics_topic2vec.length)? topics_topic2vec[i]['topic_words']: "";
                    results.push({"TF-IDF": tf_idf_term, "collocation": collocation,
                                  "topic2vec": topic2vec});
                }
                done(results);
            },
            totalNumber: topics_tf_idf.length,
            pageSize: 20,
            showNavigator: true,
            formatNavigator: '<span style="color: #f00"><%= currentPage %></span>/<%= totalPage %> pages, <%= totalNumber %> topics',
            position: 'top',
            showGoInput: false,
            showGoButton: false,
            callback: function (topics, pagination) {
                docTable.find('tbody').empty();
                for (let topic of topics) {
                    let row = $('<tr></tr>');
                    // TF-IDF term
                    row.append( $('<td>' + topic['TF-IDF'] + '</td>'));
                    // Collocation
                    row.append( $('<td>' + topic['collocation'] + '</td>'));
                    //
                    row.append( $('<td>' + topic['topic2vec'] + '</td>'));
                    docTable.find('tbody').append(row);
                }
            }
        });
        return pagination;
    }

    // topic_words = topic_words.slice(0, 10);
    function _createUI(){
        $('#topic_list_view').empty();
        const topics_tf_idf = cluster['TopicWords_by_TF-IDF'];
        const topics_collocation = cluster['TopicWords_by_Collocations'];
        const topics_topic2vec = cluster['TopicWords_by_CTF-IDF'];
        const container = $('<div class="container"></div>')
        // heading displays a number of clustered articles

        const heading = $('<div class="row p-3 mt-3">' +
            '<div class="col"><span class="h5">Cluster #' + cluster_no + ' has ' + '<a href="#">'
                + cluster['NumDocs'] + ' articles</a> and includes ' + topics_tf_idf.length + ' topics.</span></div>' +
            '</div>');
        // Bind the event to a link
        heading.find('a').click(function(){
            // Display all documents of the cluster
            const doc_list_view = new DocumentListView(cluster_docs, []);
            return false;
        });
        container.append(heading);
        // Create an auto-complete bar to look for
        const tags = topics_tf_idf.map(t => t['topic_words']);
        const tag_div = $('<div class="ui-widget"><label for="topic_tags" class="m-2">' +
            'Topics: </label><input></div>');
        // Bind the autocomplete to input
        tag_div.find('input').autocomplete({
            source: tags
        });
        // Search Button
        const search_btn = $('<button>Search</button>');
        search_btn.button();
        search_btn.on("click", function(event){
            const select_topic = tag_div.find('input').val();
            const topic = topics_tf_idf.find(t => t['topic_words'] === select_topic);
            // Get the articles related to the selected topic.
            const topic_docs = Utility.get_documents_by_doc_ids(topic['doc_ids']);
            const doc_list_view = new DocumentListView(topic_docs, [topic['topic_words']]);
        });
        // Add search btn to
        tag_div.append(search_btn);
        const tag_col = $('<div class="row p-3"><div class="col"></div></div>');
        tag_col.find(".col").append(tag_div);
        container.append(tag_col);
        // Create a table for topics
        const table = $('<table class="table table-sm"><thead><tr>' +
            '<th>Topics by TF-IDF</th>' +
            '<th>Topics by Collocations</th>' +
            '<th>Topics by Topic2Vec</th>' +
            '</tr></thead><tbody></tbody></table>');
        const pagination = createPagination(table, topics_tf_idf, topics_collocation, topics_topic2vec);
        const table_col = $('<div class="col"></div>');
        table_col.append(table);
        table_col.append(pagination);
        container.append($('<div class="row p-3"></div>').append(table_col));
        $('#topic_list_view').append(container);
        // By default, it clear all document list view.
        const doc_list_view = new DocumentListView(cluster_docs, []);
    }

    _createUI()
}
