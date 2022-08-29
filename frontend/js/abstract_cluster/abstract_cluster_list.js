// Displays a list view of all article clusters
function AbstractClusterList(corpus_data, cluster_data, abstract_clusters) {
    const common_terms = abstract_clusters[0]['common_terms'];

    // Create a common term div
    function createCommonTermDiv(){
        const div = $('<div></div>');
        div.append($('<div class="fw-bold">Common Terms:</div>'));
        const term_list = [];
        for(const term of common_terms){
            term_list.push(term);
        }
        term_list.sort();
        const term_div = $('<div class="small p-1"></div>');
        let count = 0;
        for(const term of term_list){
            if(count < term_list.length-1){
                term_div.append($('<span>' + term+', </span>'));
            }else{
                term_div.append($('<span>' + term+' </span>'));
            }
            count = count + 1;
        }
        div.append(term_div);
        return div;
    }

    // Create a pagination to show the article clusters
    function createPagination(container) {
        // Create the pagination
        const pagination = $('<div></div>');
        // Create the list
        const list_view = $('<table class="table table-sm small"></table>');
        // Pagination
        pagination.pagination({
            dataSource: function (done) {
                let result = [];
                for (let i = 0; i < abstract_clusters.length; i++) {
                    result.push(abstract_clusters[i]);
                }
                done(result);
            },
            totalNumber: abstract_clusters.length,
            pageSize: 10,
            showPageNumbers: false,
            showNavigator: true,
            formatNavigator: '<span style="color: #f00"><%= currentPage %></span>/<%= totalPage %> pages, ' +
                '<%= totalNumber %> abstract clusters',
            callback: function (clusters, pagination) {
                list_view.empty();
                // Add each keyword cluster
                list_view.append($('<thead><tr>' +
                    '<th>Abstract Cluster (Score)</th>' +
                    '<th>Abstract Number </th>' +
                    '<th>Terms</th>' +
                    '</tr></thead>'));
                const table_body = $('<tbody></tbody>');
                for (let i = 0; i < clusters.length; i++) {
                    const cluster = clusters[i];
                    table_body.append(createArticleClusterView(cluster))
                }
                list_view.append(table_body);
            }
        });
        container.append(list_view);
        if(abstract_clusters.length > 5){
            container.append(pagination);
        }

    }

    // Create a view to display a keyword cluster
    function createArticleClusterView(abstract_cluster) {
        const cluster_no = abstract_cluster['cluster'];
        const doc_ids = abstract_cluster['doc_ids'];
        const color = get_color(abstract_cluster);
        const article_cluster_view = $('<tr></tr>');
        const score = parseFloat(abstract_cluster['score']).toFixed(2);
        // Create a button to show the article cluster
        const btn = $('<button type="button" class="btn btn-link btn-sm" style="color:' + color + '">' +
            cluster_no  + ' <span style="color:' + color+'">(' + score +')</span></button>');
        btn.button();
        btn.click(function (event) {
            const doc_list = new ClusterDocList(cluster_no, corpus_data, abstract_clusters, color);
            // Highlight the dots of a specific keyword cluster
            const chart = new ScatterGraph(corpus_data, cluster_data, cluster_no);
        });
        article_cluster_view.append($('<td></td>').append(btn));
        if (score < 0.0) {
            // Add a col to display score
            article_cluster_view.append($('<td><div class="mt-2">' + doc_ids.length + '</div></td>'));
        } else {
            // Add a col to display score
            article_cluster_view.append($('<td><div class="mt-2">' + doc_ids.length + '</div></td>'));
        }

        const term_div = $('<div></div>');
        const cluster_terms = abstract_cluster['unique_terms'];
        // Display individual frequent terms
        for (let i=0; i<cluster_terms.length ; i++) {
            const term = cluster_terms[i]['term'];
            if(i < cluster_terms.length -1){
                const term_view = $('<span> ' + term + ', </span>');
                term_div.append(term_view);
            }else{
                const term_view = $('<span> ' + term + ' </span>');
                term_div.append(term_view);
            }
        }
        article_cluster_view.append($('<td></td>').append(term_div));

        return article_cluster_view;
    }

    function createUI() {
        $('#abstract_cluster_list').empty();
        const container = $('<div class="container-sm"></div>');
        container.append(createCommonTermDiv());
        createPagination(container);
        $('#abstract_cluster_list').append(container);
        $('#abstract_cluster_term_list').empty();
        $('#abstract_cluster_header').empty();
        $('#abstract_cluster_doc_list').empty();
    }

    createUI();

}
