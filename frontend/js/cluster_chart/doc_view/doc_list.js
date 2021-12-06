// Display a list of research articles
function DocList(cluster_docs, cluster_topics, selected_topics, corpus_key_phrases) {

    // Create a pagination to show the documents
    function createPagination(docTable) {
        // Create the table
        let pagination = $("<div></div>");
        // Pagination
        pagination.pagination({
            dataSource: function (done) {
                let result = [];
                for (let i = 0; i < cluster_docs.length; i++) {
                    result.push(cluster_docs[i]);
                }
                done(result);
            },
            totalNumber: cluster_docs.length,
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
                    // console.log(doc);
                    const row = $('<tr class="d-flex"></tr>');
                    // Add the title
                    const col = $('<td class="col"></td>');
                    const doc_key_phrases = corpus_key_phrases.find(c => c['DocId'] === doc['DocId'])['key-phrases'];
                    const doc_view = new DocView(doc, doc_key_phrases, cluster_topics, selected_topics);
                    col.append(doc_view.get_container());
                    row.append(col);
                    docTable.find('tbody').append(row);
                }
            }
        });
        return pagination;
    }


    function _createUI() {
        $('#doc_list').empty();
        // A list of cluster documents
        const doc_table = $('<table class="table table-borderless">' +
            '<tbody></tbody></table>');
        const pagination = createPagination(doc_table);
        // Add the table to display the list of documents.
        $('#doc_list').append(pagination);
        $('#doc_list').append(doc_table);
    }

    _createUI();
}
