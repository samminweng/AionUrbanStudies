// Display a list of topic words for a cluster
function TopicListView(cluster_no, topic_words, documents){

    function _createUI(){
        $('#topic_list_view').empty();
        // Go through each topic word to sum up all the documents
        let container = $('<div class="container p-3"></div>');
        container.append($('<div class="row"><div class="col h5">Cluster ' + cluster_no + ' has ' + documents.length
            + ' articles.</div></div>'));
        // Create a table for topics
        const table = $('<table class="table table-sm"><thead><tr>' +
            '<th>Top 10 Topic</th>' +
            '<th>Number of Articles</th>' +
            '</tr></thead><tbody></tbody></table>');
        const tbody = table.find('tbody');
        // Find out how many each topic appears in the documents
        for(const topic_word of topic_words){
            const topic = topic_word['topic'];
            const doc_ids = topic_word['doc_ids'];
            const topic_docs = documents.filter(d => doc_ids.includes(parseInt(d['DocId'])));
            const row = $('<tr></tr>');
            const topic_btn = $('<button type="button" class="btn btn-link">' + topic + '</button>');
            // Add btn click event
            topic_btn.on("click", function(){
                const doc_list_view = DocumentListView(cluster_no, topic, topic_docs);
            });
            const topic_div = $('<th scope="row"></th>');
            topic_div.append(topic_btn);
            const doc_div = $('<td>' + topic_docs.length + '</td>');
            row.append(topic_div);
            row.append(doc_div);
            tbody.append(row);
        }
        // By default, it clear all document list view.
        $('#document_list_view').empty();

        container.append(table);
        $('#topic_list_view').append(container);
    }

    _createUI()
}
