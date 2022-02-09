// Display a list of research articles
function DocList(docs, selected_terms, head_text) {
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
            showNavigator: true,
            formatNavigator: '<span style="color: #f00"><%= currentPage %></span>/<%= totalPage %> pages',
            header: '<div class="mb-3"><span class="fw-bold"><%= totalNumber %> papers </span> about ' +
                '<span class="text-capitalize">' + head_text + '</span></div>',
            position: 'top',
            callback: function (docs, pagination) {
                docTable.find('tbody').empty();
                for (let i = 0; i < docs.length; i++) {
                    const doc = docs[i];
                    console.log(doc);
                    const row = $('<tr class="d-flex"></tr>');
                    // Add the title
                    const col = $('<td class="col"></td>');
                    const doc_view = new DocView(doc, selected_terms);
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
        const container = $('<div></div>');
        // A list of cluster documents
        const doc_table = $('<table class="table table-borderless">' +
            '<tbody></tbody></table>');
        const pagination = createPagination(doc_table);
        // Add the table to display the list of documents.
        container.append(pagination);
        container.append(doc_table);
        $('#doc_list').append(container);
        document.getElementById('doc_list').scrollIntoView({behavior: "smooth", block: "start", inline: "nearest"});
    }

    _createUI();
}
