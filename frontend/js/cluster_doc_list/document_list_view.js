// Display a list of research articles
function DocumentListView(cluster_no, documents, topic) {
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
            // showGoInput: true,
            // showGoButton: true,
            callback: function (documents, pagination) {
                docTable.find('tbody').empty();
                for (let i =0; i< documents.length; i++) {
                    const doc = documents[i];
                    const row = $('<tr class="d-flex"></tr>');
                    // Add the number
                    row.append($('<td style="width:5%">' + (i + 1) + '</td>'))
                    // Add the year
                    row.append($('<td style="width:5%">' + doc['Year'] + '</td>'));
                    // Add the title
                    const col = $('<td style="width:90%"></td>');
                    let textView = new TextView(doc, topic);
                    col.append(textView.get_container());
                    row.append(col);
                    docTable.find('tbody').append(row);
                }
            }
        });
        return pagination;
    }

    function _createUI() {
        $('#document_list_view').empty();
        const container = $('<div class="container p-3"></div>');
        // If the topic is passed to the doc list view, display a summary

        const heading = $('<div class="h5 mb-3">Cluster #' + cluster_no  + ' has ' + documents.length + ' articles.</div>');
        container.append($('<div class="row p-3"><div class="col"></div></div>').find(".col").append(heading));

        // A list of cluster documents
        const documentTable = $('<table class="table table-striped">' +
            '<thead class="thead-light">' +
            '<tr class="d-flex">' +
            '    <th style="width:5%">No</th>' +
            '    <th style="width:5%">Year</th>' +
            '    <th style="width:90%">Articles</th>' +
            '</tr>' +
            '</thead>' +
            '<tbody></tbody></table>');
        const pagination = createPagination(documentTable);
        // Add the pagination
        // container.append($('<div class="row p-3"><div class="col"></div></div>').find(".col").append(pagination));
        container.append($('<div class="row p-3"><div class="col"></div></div>').find(".col").append(documentTable));
        $('#document_list_view').append(container);

    }

    _createUI();
}
