// Display a list of research articles
function DocList(cluster_data, docs, selected_term) {
    const cluster_no = cluster_data['Cluster'];
    const cluster_num_docs = cluster_data['NumDocs'];
    function createListHeading(){
        $('#doc_list_heading').empty();
        const container = $('<div class="container"></div>');
        let heading_text = 'Cluster #' + cluster_no;
        if (selected_term !== null) {
            if ('topic' in selected_term){
                heading_text += ' has ' + docs.length + " papers" +
                    " about <span class='search_terms'> " + selected_term['topic'] + "</span>";
            }else if('group' in selected_term){
                // console.log(selected_term);
                const top_key_phrases = selected_term['key-phrases'].slice(0, 3);
                heading_text += ' has ' + selected_term['DocIds'].length + " papers " +
                    " about <span class='search_terms'> " + top_key_phrases.join(', ') +
                    " and other " + selected_term['key-phrases'].length + " key phrases</span>";
            }
        } else {
            heading_text += ' has ' + docs.length + ' articles ';
        }
        // Display a summary on the heading
        const heading = $('<div class="h5 mb-3">' + heading_text + ' </div>');
        container.append($('<div class="row p-3"><div class="col"></div></div>').find(".col").append(heading));
        $('#doc_list_heading').append(container);
    }


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
            pageSize: docs.length,
            showNavigator: false,
            formatNavigator: '<span style="color: #f00"><%= currentPage %></span>/<%= totalPage %> pages, <%= totalNumber %> articles',
            position: 'top',
            callback: function (results, pagination) {
                docTable.find('tbody').empty();
                for (let i = 0; i < results.length; i++) {
                    const doc = results[i];
                    const row = $('<tr class="d-flex"></tr>');
                    // Add the year
                    row.append($('<td class="col-1">' + doc['Year'] + '</td>'));
                    // Add the title
                    const col = $('<td class="col-11"></td>');
                    const doc_key_phrases = cluster_data['KeyPhrases'];
                    const doc_view = new DocView(doc, selected_term, doc_key_phrases);
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
        createListHeading();

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
