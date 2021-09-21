// Create a list view to display all the cluster information
function ClusterListView(total_clusters) {

    // Container
    function createPagination(tbody, rank) {
        const clusters = Utility.cluster_topic_words_dict[total_clusters];
        // Create the table
        let pagination = $('<div></div>');
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
                        const topicListView = new TopicListView(cluster_no, topic_words, documents);
                    });
                    const cluster_div = $('<th scope="row" style="width:15%"></th>');
                    cluster_div.append(cluster_btn);
                    row.append(cluster_div);
                    // Number of articles
                    row.append($('<td style="width:15%">' + cluster['NumDocs'] + '</td>'));
                    // Topic words
                    // Display top 5 topic words extracted from chi-square
                    const topic_words = cluster['TopWords_'+rank].map(w => w['collocation']).slice(0, 5);
                    const topic_words_div = $('<td style="width:70%"><span>' + topic_words.join("; ") + '</span></div>');
                    row.append(topic_words_div);
                    tbody.append(row);
                }
            }
        });
        return pagination;
    }


    // Create a table
    function create_cluster_summary_table(rank) {
        const table = $('<table class="table"><thead><tr>' +
            '<th></th>' +
            '<th># Articles</th>' +
            '<th>Most Relevant Topics' +
            '</th>' +
            '</tr></thead><tbody></tbody></table>');
        // Set the ranke
        table.find('select').val(rank);
        const tbody = table.find('tbody');
        const pagination = createPagination(tbody, rank);
        return {table: table, pagination: pagination};
    }

    function _createUI() {
        // Create a summary for clusters
        $('#cluster_list_view').empty();
        let container = $('<div class="container p-3"></div>')
        const total_document = Utility.corpus_data.length;
        const overview = $('<div class="col">' + total_clusters + ' Clusters extracted from ' + total_document +
            ' articles. ' +
            '    <span class="text-end">Topics are ranked by' +
            '    <select name="rank">' +
            '      <option value="chi">Chi-Square</option>' +
            '      <option value="pmi">PMI</option>' +
            '      <option value="likelihood">Likelihood</option>' +
            '    </select></span>' +
            '</div>');
        container.append($('<div class="row"><div class="col"></div></div>').find('.col').append(overview));
        const {table, pagination} = create_cluster_summary_table('chi');
        container.append($('<div class="row"><div class="col cluster-table"></div></div>').find('.cluster-table').append(table));
        container.append($('<div class="row"><div class="col cluster-pagination"></div></div>').find('.cluster-pagination').append(pagination));

        $('#cluster_list_view').append(container);
        // Create a tab to display the clustered documents
        const select_rank = container.find('select');
        select_rank.on("change", function(){
            // alert("rank" + this.value);
            const rank = this.value;
            // Create a new table and pagination
            const {table, pagination} = create_cluster_summary_table(rank);
            $('#cluster_list_view').find('.cluster-table').empty();
            $('#cluster_list_view').find('.cluster-table').append(table);
            $('#cluster_list_view').find('.cluster-pagination').empty();
            $('#cluster_list_view').find('.cluster-pagination').append(pagination);
        });
    }

    _createUI();

}
