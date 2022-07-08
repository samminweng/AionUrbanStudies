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
        $('#article_cluster_header').empty();
        let heading_text = "";
        if(selected_term !== null){
            heading_text = "about " + selected_term;
        }
        const container = $('<div class="container-sm"></div>');
        const row = $('<div class="row"></div>');
        const col = $('<div class="col-3"></div>');
        // Add heading
        col.append($('<span class="fw-bold"> ' + docs.length + ' abstract </span>' +
            '<span>' + heading_text + '</span>'));
        row.append(col);
        const sort_col = $('<div class="col-6"></div>');
        // Add sort by button
        sort_col.append($('<span>Sort by </span>'));
        // Sort by citation
        sort_col.append($('<span class="form-check-sm form-check-inline">' +
            '<input class="form-check-input" type="radio" name="sort-btn" value="citation" checked>' +
            '<label class="form-check-label"> Citation </label>' +
            '</span>'));
        // Sort by year
        sort_col.append($('<span class="form-check-sm form-check-inline">' +
            '<input class="form-check-input" type="radio" name="sort-btn" value="year">' +
            '<label class="form-check-label"> Year </label>' +
            '</span>'));
        row.append(sort_col);
        container.append(row);
        // Define onclick event
        sort_col.find('input[name="sort-btn"]').change(function(){
            sort_col.find('input[name="sort-btn"]').prop('checked', false);
            if(this.value === "year"){
                sort_docs("Year");
                sort_col.find('input[value="year"]').prop("checked", true);
            }else{
                sort_docs('Cited by');
                sort_col.find('input[value="citation"]').prop("checked", true);
            }
            createDocList();
            // _createUI();
        });
        $('#article_cluster_header').append(container);
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
                    const doc_view = new DocView(doc, selected_term);
                    docTable.append(doc_view.get_container());
                }
            }
        });
        return pagination;
    }
    // Create a list of docs
    function createDocList(){
        $('#article_cluster_doc_list').empty();
        const container = $('<div></div>');
        const doc_table = $('<div class="card-group"></div>');
        const pagination = createPagination(doc_table);
        // Add the table to display the list of documents.
        container.append(pagination);
        container.append(doc_table);
        $('#article_cluster_doc_list').append(container);
    }

    // Create header and doc list
    function _createUI() {
        createHeading();
        createDocList();
    }

    _createUI();

}
