// Displays a list view of all article clusters
function ArticleClusterList(corpus_data, cluster_data, article_clusters) {
    const common_terms = collectCommonTerms(article_clusters);
    // console.log(common_terms);
    // Get the cluster color by group number
    function get_color(article_cluster) {
        const cluster_no = article_cluster['Cluster'];
        const group_no = article_cluster['Group'];
        // Get the group colors < group_no
        let index = 0;
        for (let i = 1; i < group_no; i++) {
            index += group_color_plates[i].length;
        }
        let color_index = cluster_no - index - 1;
        return group_color_plates[group_no][color_index];
    }

    // Collect the common terms from top 10 freq terms
    function collectCommonTerms(clusters){
        let common_terms = new Map();
        for(const cluster of clusters){
            const freq_terms = cluster['FreqTerms'].slice(0, 10);
            for(const freq_term of freq_terms){
                const term = freq_term['term'];
                let count = 1;
                if(common_terms.has(term)){
                    count = common_terms.get(term) + 1;
                }
                common_terms.set(term, count);
            }
        }
        return common_terms;
    }

    // Create a common term div
    function createCommonTermDiv(){
        const div = $('<div></div>');
        div.append($('<div class="fw-bold">Common Terms:</div>'));
        const term_list = [];
        for(const [term, count] of common_terms.entries()){
            if(count > 1){
                term_list.push(term);
            }
        }
        term_list.sort();
        const term_div = $('<div></div>');
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
                for (let i = 0; i < article_clusters.length; i++) {
                    result.push(article_clusters[i]);
                }
                done(result);
            },
            totalNumber: article_clusters.length,
            pageSize: 10,
            showPageNumbers: false,
            showNavigator: true,
            formatNavigator: '<span style="color: #f00"><%= currentPage %></span>/<%= totalPage %> pages, ' +
                '<%= totalNumber %> article clusters',
            callback: function (clusters, pagination) {
                list_view.empty();
                // Add each keyword cluster
                list_view.append($('<thead><tr>' +
                    '<th>Abstract Cluster (Score)</th>' +
                    '<th>Abstract Number </th>' +
                    '<th>Frequent Terms</th>' +
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
        if(article_clusters.length > 5){
            container.append(pagination);
        }

    }

    // Create a view to display a keyword cluster
    function createArticleClusterView(article_cluster) {
        const cluster_no = article_cluster['Cluster'];
        const doc_ids = article_cluster['DocIds'];
        const color = get_color(article_cluster);
        const article_cluster_view = $('<tr></tr>');
        const score = parseFloat(article_cluster['Score']).toFixed(2);
        // Create a button to show the article cluster
        const btn = $('<button type="button" class="btn btn-link btn-sm" style="color:' + color + '">' +
            cluster_no  + ' <span style="color:' + color+'">(' + score +')</span></button>');
        btn.button();
        btn.click(function (event) {
            const doc_list = new ClusterDocList(cluster_no, corpus_data, article_clusters, color, common_terms);
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
        const terms = article_cluster['FreqTerms'];
        let count = 0;
        // Display top 10 key phrases
        for (let i=0; i<terms.length && count<10; i++) {
            const term = terms[i];
            if(common_terms.has(term['term']) && common_terms.get(term['term'])>1){
                continue;
            }
            if(count === 10){
                const term_view = $('<span> ' + term['term'] + ' </span>');
                term_div.append(term_view);
            }else{
                const term_view = $('<span> ' + term['term'] + ', </span>');
                term_div.append(term_view);
            }
            count = count + 1;
        }
        article_cluster_view.append($('<td></td>').append(term_div));

        return article_cluster_view;
    }


    // console.log(article_clusters);

    function createUI() {

        $('#article_cluster_list').empty();
        const container = $('<div class="container-sm small"></div>');
        container.append(createCommonTermDiv());
        createPagination(container);
        $('#article_cluster_list').append(container);
        $('#article_cluster_term_list').empty();
        $('#article_cluster_doc_list').empty();
    }

    createUI();

}
