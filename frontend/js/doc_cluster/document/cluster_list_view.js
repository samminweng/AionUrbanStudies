// Create a list view to display all the cluster information
function ClusterListView(total_clusters) {
    // Create a table
    function create_cluster_summary_table(total_document) {
        const table = $('<table class="table"><thead><tr>' +
            '<th></th>' +
            '<th># Articles</th>' +
            '<th>Most Relevant Topics</th>' +
            '</tr></thead><tbody></tbody></table>');
        const tbody = table.find('tbody');
        // Go through each cluster
        for (let cluster_no = 0; cluster_no < total_clusters; cluster_no++) {
            const row = $('<tr></tr>');
            const cluster_btn = $('<button type="button" class="btn btn-link">Cluster ' + cluster_no  + '</button>');
            // Onclick event to display the topic words results
            cluster_btn.on("click", function(){
                const {topic_words, documents} = Utility.get_cluster_documents(total_clusters, cluster_no);
                const topicListView = new TopicListView(cluster_no, topic_words, documents);
            })


            const cluster_div = $('<th scope="row" style="width:15%"></th>');
            cluster_div.append(cluster_btn);
            row.append(cluster_div);
            const dict = Utility.cluster_topic_words_dict[total_clusters][cluster_no];
            row.append($('<td style="width:15%">' + dict['NumDocs'] + '</td>'));
            // Math.round(parseInt(dict['NumDocs']) / total_document * 100)
            // Display topic words
            const topic_words = dict['TopWords'].slice(0, 10);
            const topic_words_div = $('<td style="width:70%"><span>' + topic_words.join("; ") + '</span></div>');
            row.append(topic_words_div);
            tbody.append(row);
        }
        return table;
    }

    function _createUI() {
        // Create a summary for clusters
        $('#cluster_list_view').empty();
        let container = $('<div class="container p-3"></div>')
        const total_document = Utility.corpus_data.length;
        const overview = $('<div class="col h5">' + total_clusters + ' Clusters extracted from ' + total_document + ' articles.</div>');
        container.append($('<div class="row"><div class="col"></div></div>').find('.col').append(overview));
        const table = create_cluster_summary_table(total_document);
        container.append($('<div class="row"><div class="col"></div></div>').find('.col').append(table));
        $('#cluster_list_view').append(container);
        // Create a tab to display the clustered documents
        // create_cluster_tabs();
    }

    _createUI();

}
