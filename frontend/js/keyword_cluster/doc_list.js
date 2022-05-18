// Create a doc list view including pagination and doc
function DocList(docs, keywords, color_no){

    // Create a pagination to show the documents
    function displayDocList() {
        // Create the table
        let pagination = $("<div class='mb-3'></div>");
        // Add the table header
        const doc_list = $('<ul class="list-group list-group-flush"></ul>');
        // // Pagination
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
            formatNavigator: '<span style="color: #f00"><%= currentPage %></span>/<%= totalPage %> pages, ' +
                '<%= totalNumber %> articles',
            position: 'top',
            className: 'paginationjs-theme-blue paginationjs-small',
            // showGoInput: true,
            // showGoButton: true,
            callback: function (_docs, pagination) {
                // console.log(_docs);
                doc_list.empty();
                for (let i = 0; i < _docs.length; i++) {
                    const doc = _docs[i];
                    const doc_view = new DocView(doc, keywords, color_no);
                    doc_list.append(doc_view.get_container());
                }
            }
        });
        pagination.append(doc_list);
        return pagination;
    }


    function createUI(){
        $('#doc_list_view').empty();
        $('#doc_list_view').append(displayDocList());
    }

    createUI();
}
