// Create a list view to display all the cluster information
function ClusterListView(total_clusters) {

    // Container
    function createPagination(rank) {
        const tbody = $('#cluster_table').find('tbody');
        const clusters = Utility.cluster_topic_words_dict[total_clusters];
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
                    // Onclick event to display the topic words results
                    cluster_btn.on("click", function () {
                        const documents = Utility.get_cluster_documents(total_clusters, cluster_no);
                        const topic_words = Utility.get_cluster_topic_words(total_clusters, cluster_no, rank);
                        console.log(topic_words);
                        const topicListView = new TopicListView(cluster_no, topic_words, documents);
                    });
                    const cluster_div = $('<th scope="row" style="width:15%"></th>');
                    cluster_div.append(cluster_btn);
                    row.append(cluster_div);
                    // Number of articles
                    row.append($('<td style="width:15%">' + cluster['NumDocs'] + '</td>'));
                    // Topic words
                    // Display top 5 topic words extracted from chi-square
                    const topic_words = cluster['Topic_Words_'+rank].map(w => w['collocation']).slice(0, 5);
                    const topic_words_div = $('<td style="width:70%"><span>' + topic_words.join("; ") + '</span></div>');
                    row.append(topic_words_div);
                    tbody.append(row);
                }
            }
        });

    }

    function _createUI() {
        // Update the overview
        $('#cluster_overview').text(total_clusters + " clusters are extracted from 600 articles " +
            "using BERT-based Sentence Transformer + KMeans clustering technique.")
        // Create a pagination and display topic words by Chi-square test
        createPagination('likelihood');
        // // Create a tab to display the clustered documents
        const select_rank = $('#cluster_table').find('select');
        select_rank.on("change", function(){
            // Clear the detailed cluster-topic/document view
            $('#topic_list_view').empty();
            $('#document_list_view').empty();
            // alert("rank" + this.value);
            const rank = this.value;
            // Create a new table and pagination
            createPagination(rank);
        });
    }

    _createUI();

}
