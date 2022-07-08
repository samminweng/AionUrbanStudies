// Display a list of research articles
function DocList(docs, cluster, selected_term) {
    // console.log(cluster);
    sort_docs('Cited by'); // Sort citation by default
    function sort_docs(name){
        // Sort docs by citations
        docs.sort((a, b) =>{
            if(a[name] === b[name]){
                return 0;
            }
            // Null values
            if(b[name] === null){
                return -1;
            }
            if(a[name] === null){
                return 1;
            }
            // Non-null values
            if(a[name] && b[name]){
                if(b[name] > a[name]){
                    return 1;
                }else{
                    return -1;
                }
            }
        });
    }

    // Create a heading to display the number of articles and sort function
    function createHeading(){
        let heading_text = "";
        if(selected_term !== null){
            heading_text = "about " + selected_term;
        }
        const view = $('<div></div>');
        const container = $('<div class="row"></div>')
        // Add heading
        container.append($('<div class="col"><span class="fw-bold"> ' + docs.length + ' abstract </span>' +
            '<span>' + heading_text + '</span></div>'));
        // Add sort by button
        const sort_by_div = $('<div>Sort by </div>');
        // Sort by citation
        sort_by_div.append($('<div class="form-check form-check-inline">' +
            '<input class="form-check-input" type="radio" name="sort-btn" value="citation" checked>' +
            '<label class="form-check-label"> Citation </label>' +
            '</div>'));
        // Sort by year
        sort_by_div.append($('<div class="form-check form-check-inline">' +
            '<input class="form-check-input" type="radio" name="sort-btn" value="year">' +
            '<label class="form-check-label"> Year </label>' +
            '</div>'));
        container.append($('<div class="col"></div>').append(sort_by_div));
        view.append(container);

        // Define onclick event
        sort_by_div.find('input[name="sort-btn"]').change(function(){
            if(this.value === "year"){
                sort_docs("Year");
            }else{
                sort_docs('Cited by');
            }
            _createUI();
        });
        return view;
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
            pageSize: 3,
            showNavigator: true,
            formatNavigator: '<span style="color: #f00"><%= currentPage %></span>/<%= totalPage %> pages',
            callback: function (docs, pagination) {
                // docTable.find('tbody').empty();
                docTable.empty();
                for (let i = 0; i < docs.length; i++) {
                    const doc = docs[i];
                    // console.log(doc);
                    // const row = $('<tr class="d-flex"></tr>');
                    // // Add the title
                    // const col = $('<td class="col"></td>');
                    const doc_view = new DocView(doc, selected_term);
                    // col.append(doc_view.get_container());
                    // row.append(col);
                    // docTable.find('tbody').append(doc_view.get_container());
                    docTable.append(doc_view.get_container());
                }
            }
        });
        return pagination;
    }

    function _createUI() {
        $('#article_cluster_doc_list').empty();
        const container = $('<div class="small"></div>');
        // A list of cluster documents
        // const doc_table = $('<table class="table table-borderless table-sm">' +
        //     '<tbody></tbody></table>');
        const doc_table = $('<div class="card-group"></div>');
        const pagination = createPagination(doc_table);
        // Add the table to display the list of documents.
        container.append(createHeading());
        container.append(pagination);
        container.append(doc_table);
        $('#article_cluster_doc_list').append(container);
    }

    _createUI();

}
