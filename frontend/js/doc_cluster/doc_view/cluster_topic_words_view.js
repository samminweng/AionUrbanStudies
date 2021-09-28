// Display a list of topic words for a cluster
function ClusterTopicWordsView(cluster){
    const cluster_no = cluster['Cluster'];

    // topic_words = topic_words.slice(0, 10);
    function _createUI(){
        $('#topic_list_view').empty();
        // Go through each topic word to sum up all the documents
        let container = $('<div class="container p-3"></div>');
        container.append($('<div class="row"><div class="col h5">Cluster #' + cluster_no + ' has ' + cluster['NumDocs']
            + ' articles.</div></div>'));
        // Create a table for topics
        const table = $('<table class="table table-sm"><thead><tr>' +
            '<th>Topics by TF-IDF</th>' +
            '<th>Topics by Collocations</th>' +
            '<th>Topics by Topic2Vec</th>' +
            '</tr></thead><tbody></tbody></table>');
        const tbody = table.find('tbody');
        const topics_tf_idf = cluster['TopicWords_by_TF-IDF'].slice(0, 30);
        const topics_collocation = cluster['TopicWords_by_Collocations'].slice(0, 30);
        const topics_topic2vec = cluster['TopicWords_by_CTF-IDF'].slice(0, 30);
        const max_topics = Math.min(topics_tf_idf.length, topics_collocation.length, topics_topic2vec.length);
        // Find out how many each topic appears in the documents
        for(let i =0; i< max_topics; i++){
            const row = $('<tr></tr>');
            const topic = topics_tf_idf[i];
            const link = $('<button type="button" class="btn btn-link">' + topic['topic_words'] +
                ' (' + topic['doc_ids'].length + ')</button>');
            // Add event to link
            link.on("click", function(){
               const topic_docs = Utility.get_documents_by_doc_ids(topic['doc_ids']);
               const doc_list_view = new DocumentListView(topic_docs, [topic['topic_words']]);
            });
            row.append($('<td></td>').append(link));
            row.append($('<td>' + topics_collocation[i]['topic_words'] + '</td>'));
            row.append($('<td>' + topics_topic2vec[i]['topic_words'] + '</td>'));
            tbody.append(row);
        }
        // By default, it clear all document list view.
        const cluster_docs = Utility.get_documents_by_doc_ids(cluster['DocIds']);
        const doc_list_view = new DocumentListView(cluster_docs, []);
        container.append(table);
        $('#topic_list_view').append(container);



    }

    _createUI()
}
