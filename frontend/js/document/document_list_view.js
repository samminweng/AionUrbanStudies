// Display a list of research articles for key term 1 and key term 2.
function DocumentListView(key_term_1, key_term_2, collocation_data, corpus_data) {
    const collocation_1 = collocation_data.find(({Collocation}) => Collocation === key_term_1);
    const collocation_2 = collocation_data.find(({Collocation}) => Collocation === key_term_2);
    const documents_1 = new Set(Utility.collect_documents(collocation_1, corpus_data));
    const documents_2 = new Set(Utility.collect_documents(collocation_2, corpus_data));
    // Use the set intersection to find
    const document_set = new Set([...documents_1].filter(doc => documents_2.has(doc)));
    const documents = [...document_set];
    console.log(documents);
    // Container
    function createPagination(docTable){
        // Create the table
        let pagination = $("<div></div>");
        // Pagination
        pagination.pagination({
            dataSource: function (done) {
                let result = [];
                for (let i = 1; i <= documents.length; i++) {
                    result.push(documents[i - 1]);
                }
                done(result);
            },
            totalNumber: documents.length,
            pageSize: 5,
            showNavigator: true,
            formatNavigator: '<span style="color: #f00"><%= currentPage %></span>/<%= totalPage %> pages, <%= totalNumber %> texts',
            position: 'top',
            showGoInput: true,
            showGoButton: true,
            callback: function (documents, pagination) {
                docTable.find('tbody').empty();
                for (let document of documents) {
                    let row = $('<tr class="d-flex"></tr>');
                    // Add the year
                    let col = $('<td class="col-1">' + document['Year']+'</td>');
                    row.append(col);
                    // Add the title
                    col = $('<td class="col-11"></td>');
                    let textView = new TextView(document, [key_term_1, key_term_2]);
                    col.append(textView.get_container());
                    row.append(col);
                    docTable.find('tbody').append(row);
                }
            }
        });

        return pagination;
    }


    function _createUI() {
        $('#text_list_view').empty();
        let container = $('<div class="container"></div>');
        let documentTable = $('<table class="table table-striped">' +
            '<thead class="thead-light">' +
            '<tr class="d-flex">' +
            '    <th class="col-1">Year</th>' +
            '    <th class="col-11">Text</th>' +
            '</tr>' +
            '</thead>' +
            '<tbody></tbody></table>');

        let pagination = createPagination(documentTable);
        // // Add the pagination
        container.append(pagination);
        container.append(documentTable);

        $('#text_list_view').append(container);

        // Set the clicked terms
        let group_1 = Utility.get_group_number(key_term_1);
        $('#selected_term_1')
            .attr('class', 'keyword-group-'+ group_1)
            .text(key_term_1);
        let group_2 = Utility.get_group_number(key_term_2);
        $('#selected_term_2')
            .attr('class', 'keyword-group-'+ group_2)
            .text(key_term_2);

    }

    _createUI();
}
