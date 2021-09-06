// Display a list of texts
function DocumentListView(_documents) {
    const documents = _documents;
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
                    let textView = new TextView(document, "machine learning");
                    col.append(textView.get_container());
                    row.append(col);

                    // let doc_container = $('<div class="container"></div>');
                    // Story View
                    // let storyView;
                    // let storyPattern = storyPatterns.find(d => d['storyId'] === story['id']);
                    // // Get the story dep
                    // if (storyPattern && storyPattern['matchedPatterns'].length>0) {
                    //     let pattern = storyPattern['matchedPatterns'][0]
                    //     let matchedWords = pattern['matchedWords'];
                    //     let mode = Utility.mapPatternMode(pattern['pattern']);
                    //     // console.log(matchedWords);
                    //     storyView = new HighlightStoryView(story, matchedWords, mode);
                    // } else {
                    //     storyView = new HighlightStoryView(story, [], 'object');
                    // }
                    // storyContainer.append($('<div class="row"></div>').append(storyView.getContainer()));
                    // // Category div
                    // let categoryView = new CategoryView(story);
                    // storyContainer.append($('<div class="row"></div>').append(categoryView.getContainer()));
                    // col.append(storyContainer);
                    // row.append(col);
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



    }

    _createUI();
}
