// Create a list view to display all the cluster information
function ClusterListView(cluster_approach, cluster_topic_words_data) {
    const clusters = cluster_topic_words_data[cluster_approach].filter(c => c['Cluster'] >=0);
    const total_clusters = clusters.length;
    // Sum up all the documents that are clustered by the approach
    const total_docs = clusters.reduce((previous_value, current_value) =>{
        return previous_value + current_value['NumDocs'];
    }, 0)
    // Get the outliers detected by HDBSCAN approach
    const outliers = cluster_topic_words_data[cluster_approach].filter(c => c['Cluster'] <0);

    // Container
    function createPagination(rank) {
        const tbody = $('#cluster_table').find('tbody');

        // Create the table
        let pagination = $('#pagination');
        pagination.empty();
        // Pagination
        pagination.pagination({
            dataSource: function (done) {
                let result = [];
                for (let i = 0; i < clusters.length; i++) {
                    result.push(clusters[i]);
                }
                done(result);
            },
            totalNumber: clusters.length,
            pageSize: 5,
            showPrevious: false,
            showNext: false,
            callback: function (clusters, pagination) {
                tbody.empty();
                for (const cluster of clusters) {
                    const row = $('<tr></tr>');
                    // Cluster column
                    const cluster_no = cluster['Cluster'];
                    const cluster_btn = $('<button type="button" class="btn btn-link">Cluster ' + cluster_no + '</button>');
                    // // Onclick event to display the topic words results
                    // cluster_btn.on("click", function () {
                    //     const documents = Utility.get_cluster_documents(total_clusters, cluster_no);
                    //     const topic_words = Utility.get_cluster_topic_words(total_clusters, cluster_no, rank);
                    //     console.log(topic_words);
                    //     const topicListView = new TopicListView(cluster_no, topic_words, documents);
                    // });
                    const cluster_div = $('<th scope="row" style="width:15%"></th>');
                    cluster_div.append(cluster_btn);
                    row.append(cluster_div);
                    // Number of articles
                    row.append($('<td style="width:15%">' + cluster['NumDocs'] + '</td>'));
                    // Topic words
                    // Display top 5 topic words extracted from chi-square
                    const topic_words = cluster['TopicWords_by_'+rank].map(w => w['topic_words']).slice(0, 5);
                    const topic_words_div = $('<td style="width:70%"><span>' + topic_words.join("; ") + '</span></div>');
                    row.append(topic_words_div);
                    tbody.append(row);
                }
            }
        });

    }

    function _createUI() {
        // Update the overview
        let overview = total_clusters + ' clusters are extracted from '+ total_docs +' articles ' +
            'using BERT-based Sentence Transformer + ' + cluster_approach + ' clustering technique.';
        if(outliers.length > 0){
            overview += ' <br> ' + outliers[0]['NumDocs'] + ' articles are identified as outliers by ' +
                cluster_approach+ ' cluster technique.';
        }
        $('#cluster_overview').html(overview);
        // Create a pagination to display topic words for each cluster
        createPagination('TF-IDF');
        // Set the default ranking
        $('#rank').val('TF-IDF');
        // Create a tab to display the topic words ranked by different keyword extraction approach
        $('#rank').selectmenu({
            width: 150,
            change: function (event, data) {
                // Clear the detailed cluster-topic/document view
                $('#topic_list_view').empty();
                $('#document_list_view').empty();
                const rank = this.value;
                createPagination(rank); // Create a new table and pagination for the ranked topic words
            }
        });

    }

    _createUI();

}
