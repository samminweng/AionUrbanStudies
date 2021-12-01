// Display a list of research articles
function DocList(cluster_topics, documents, topic, corpus_key_phrases) {
    const cluster_no = cluster_topics['Cluster'];
    // Create a pagination to show the documents
    function createPagination(docTable) {
        // Create the table
        let pagination = $("<div></div>");
        // Pagination
        pagination.pagination({
            dataSource: function (done) {
                let result = [];
                for (let i = 0; i < documents.length; i++) {
                    result.push(documents[i]);
                }
                done(result);
            },
            totalNumber: documents.length,
            pageSize: documents.length,
            showNavigator: false,
            formatNavigator: '<span style="color: #f00"><%= currentPage %></span>/<%= totalPage %> pages, <%= totalNumber %> articles',
            position: 'top',
            callback: function (documents, pagination) {
                docTable.find('tbody').empty();
                for (let i = 0; i < documents.length; i++) {
                    const doc = documents[i];
                    const row = $('<tr class="d-flex"></tr>');
                    // row.append($('<td class="col-1">' + doc['DocId'] + '</td>'));
                    // Add the year
                    row.append($('<td class="col-1">' + doc['Year'] + '</td>'));
                    // Add the title
                    const col = $('<td class="col-11"></td>');
                    const doc_key_phrases = corpus_key_phrases.find(c => c['DocId'] === doc['DocId'])['key-phrases'];
                    const doc_view = new DocView(doc, topic, doc_key_phrases);
                    col.append(doc_view.get_container());
                    row.append(col);
                    docTable.find('tbody').append(row);
                }
            }
        });
        return pagination;
    }
    // Main entry
    function _createUI() {
        $('#doc_list_view').empty();
        const container = $('<div class="container"></div>');
        // A list of cluster documents
        const documentTable = $('<table class="table table-striped">' +
            '<thead>' +
            '<tr class="d-flex">' +
            '    <th class="col-1">Year</th>' +
            '    <th class="col-11">Articles</th>' +
            '</tr>' +
            '</thead>' +
            '<tbody></tbody></table>');
        const pagination = createPagination(documentTable);
        // Add the table to display the list of documents.
        container.append($('<div class="row"><div class="col"></div></div>').find(".col").append(documentTable));
        $('#doc_list_view').append(container);

    }

    _createUI();
}
