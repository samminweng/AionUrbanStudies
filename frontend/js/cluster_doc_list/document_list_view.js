// Display a list of research articles
function DocumentListView(cluster_topics, documents, topic) {
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
            // showGoInput: true,
            // showGoButton: true,
            callback: function (documents, pagination) {
                docTable.find('tbody').empty();
                for (let i = 0; i < documents.length; i++) {
                    const doc = documents[i];
                    const row = $('<tr class="d-flex"></tr>');
                    row.append($('<td class="col-1">' + doc['DocId'] + '</td>'));
                    // Add the year
                    row.append($('<td class="col-1">' + doc['Year'] + '</td>'));
                    // Add the title
                    const col = $('<td class="col-10"></td>');
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
        const container = $('<div class="container"></div>');
        const heading_text = 'Cluster #' + cluster_no + ' has ' + documents.length + ' articles '
            + ((topic !== null)? "about <span class='search_term'> " + topic['topic'] + "</span>": "");
        // Display a summary
        const heading = $('<div class="h5 mb-3">' + heading_text + '</div>');
        container.append($('<div class="row p-3"><div class="col"></div></div>').find(".col").append(heading));

        // A list of cluster documents
        const documentTable = $('<table class="table table-striped">' +
            '<thead>' +
            '<tr class="d-flex">' +
            '    <th class="col-1">DocID</th>' +
            '    <th class="col-1">Year</th>' +
            '    <th class="col-10">Articles</th>' +
            '</tr>' +
            '</thead>' +
            '<tbody></tbody></table>');
        const pagination = createPagination(documentTable);
        // Add the table to display the list of documents.
        container.append($('<div class="row"><div class="col"></div></div>').find(".col").append(documentTable));
        $('#document_list_view').append(container);

    }

    _createUI();
}
