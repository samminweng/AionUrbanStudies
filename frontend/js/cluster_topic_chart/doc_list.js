// Display a list of research articles
function DocList(docs, cluster_topics, selected_topics, cluster_docs) {

    // Create a pagination to show the documents
    function createPagination(docTable) {
        // Create the table
        let pagination = $("<div></div>");
        // Pagination
        pagination.pagination({
            dataSource: function (done) {
                let result = [];
                for (let i = 0; i < docs.length; i++) {
                    result.push(docs[i]);
                }
                done(result);
            },
            totalNumber: docs.length,
            pageSize: 5,
            showNavigator: false,
            formatNavigator: '<span style="color: #f00"><%= currentPage %></span>/<%= totalPage %> pages, <%= totalNumber %> articles',
            position: 'top',
            // showGoInput: true,
            // showGoButton: true,
            callback: function (docs, pagination) {
                docTable.find('tbody').empty();
                for (let i = 0; i < docs.length; i++) {
                    const doc = docs[i];
                    const row = $('<tr class="d-flex"></tr>');
                    // Add the title
                    const col = $('<td class="col"></td>');
                    let textView = new TextView(doc, cluster_topics, selected_topics);
                    col.append(textView.get_container());
                    row.append(col);
                    docTable.find('tbody').append(row);
                }
            }
        });
        return pagination;
    }


    function _createUI() {
        $('#topic_doc_list').empty();
        // Add a heading
        if(selected_topics){
            $('#topic_doc_list').append($('<div class="h5">'+ docs.length+ ' articles contains' +
                ' <span class="search_term">' + selected_topics['topic']+ '</span></div>'));
            // // Add the reset button to display all the cluster articles
            // const reset_btn = $('<button class="mx-1">' +
            //     '<span class="ui-icon ui-icon-closethick"></span></button>');
            // reset_btn.button();
            // reset_btn.click(function (event) {
            //     const topic_list = new TopicList(cluster_topics, cluster_docs);
            //     const doc_list = new DocList(cluster_docs, cluster_topics, null, cluster_docs);
            // });
            // $('#topic_doc_list').find('.search_term').append(reset_btn);
        }
        // A list of cluster documents
        const doc_table = $('<table class="table table-striped table-sm">' +
            '<tbody></tbody></table>');
        const pagination = createPagination(doc_table);
        // Add the table to display the list of documents.
        $('#topic_doc_list').append(pagination);
        $('#topic_doc_list').append(doc_table);
    }

    _createUI();
}
