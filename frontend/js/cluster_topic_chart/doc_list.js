// Display a list of research articles
function DocList(topic_docs, selected_topics) {

    // Create a pagination to show the documents
    function createPagination(docTable) {
        // Create the table
        let pagination = $("<div></div>");
        // Pagination
        pagination.pagination({
            dataSource: function (done) {
                let result = [];
                for (let i = 0; i < topic_docs.length; i++) {
                    result.push(topic_docs[i]);
                }
                done(result);
            },
            totalNumber: topic_docs.length,
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
                    let textView = new TextView(doc, selected_topics);
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
        const heading = $('<div class="h5">'+ topic_docs.length+ ' articles contains' +
            ' <span class="search_term">' + selected_topics['topic']+ '</span> </div>')

        // A list of cluster documents
        const doc_table = $('<table class="table table-striped table-sm">' +
            '<tbody></tbody></table>');
        const pagination = createPagination(doc_table);
        // Add the table to display the list of documents.
        $('#topic_doc_list').append(heading);
        $('#topic_doc_list').append(pagination);
        $('#topic_doc_list').append(doc_table);
    }

    _createUI();
}
