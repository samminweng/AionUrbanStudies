// Create a div to display the grouped key phrases
function KeywordClusterView(keyword_cluster, docs, color) {
    const group_no = keyword_cluster['Group'];
    const keywords = keyword_cluster['Key-phrases'];

    // Create a pagination
    // Create a pagination to show the documents
    function displayDocList() {
        // Create the table
        let pagination = $("<div></div>");
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
                console.log(_docs);
                doc_list.empty();
                for (let i = 0; i < _docs.length; i++) {
                    const doc = _docs[i];
                    const doc_view = new DocView(doc, keywords);
                    doc_list.append(doc_view.get_container());
                }
            }
        });
        pagination.append(doc_list);
        return pagination;
    }

    // Display keywords in a list
    function displayKeywordList() {
        const keyword_div = $('<div class="container-sm"></div>');
        // Get the number of rows by rounding upto the closest integer
        const num_row = Math.ceil(keywords.length / 2) + 1;
        // Add each key phrase
        for (let i = 0; i < num_row; i++) {
            // Create a new row
            const row = $('<div class="row"></div>');
            for (let j = 0; j < 2; j++) {
                const index = i * 2 + j;
                if (index < keywords.length) {
                    const keyword = keywords[index];
                    const kp_btn = $('<div class="col border-bottom">' + keyword + '</div>');
                    // // Add button
                    // kp_btn.button();
                    // // On click event
                    // kp_btn.click(function(event){
                    //     display_papers_by_key_phrase(key_phrase);
                    // });
                    row.append(kp_btn);
                }
            }
            keyword_div.append(row);
        }
        return keyword_div;
    }


    function _createUI() {
        $('#keyword_cluster_view').empty();
        // Heading
        const container = $('<div class="m-3"></div>');
        const heading = $('<div class="mb-3">' +
                          '<span class="fw-bold" style="color:' + color + '">Keyword Cluster ' + group_no + ' </span>' +
                          '</div>');
        // A list of grouped key phrases
        container.append(heading);
        container.append(displayDocList());
        // container.append(displayKeywordList());
        $('#keyword_cluster_view').append(container);
    }

    _createUI();
}
