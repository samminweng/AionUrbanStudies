// Displays a list view of all article clusters
function ArticleClusterList(corpus_data, article_clusters){
    // Optimal color pallets for 31 colors from https://medialab.github.io/iwanthue/
    const color_plates = ["#ac4876", "#42c87f", "#c34da8", "#91b23e", "#5b47a7", "#7ab65b", "#492675", "#bfa43e",
                          "#687ae2", "#c9822d", "#6697e2", "#c3552d", "#36dee6", "#cf3f80", "#43c29e", "#741a4f",
                          "#62ac6a", "#ac72d4", "#4e6b1e", "#cf8fde", "#b69d56", "#4c5ea7", "#c17a49", "#8a4087",
                          "#88341c", "#df81be", "#d85750", "#e2718d", "#9d2f39", "#e2656d", "#a1314d"];

    // Create a pagination to show the article clusters
    function createPagination(container) {
        // Create the pagination
        const pagination = $('<div></div>');
        // Create the list
        const list_view = $('<table class="table"></table>');
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
            pageSize: 5,
            showNavigator: true,
            formatNavigator: '<span style="color: #f00"><%= currentPage %></span>/<%= totalPage %> pages, ' +
            '<%= totalNumber %> article clusters',
            callback: function (clusters, pagination) {
                list_view.empty();
                // Add each keyword cluster
                list_view.append($('<thead><tr>' +
                    '<th>Cluster</th>' +
                    '<th>Article Number (Score)</th>' +
                    '<th>TFIDF Terms</th>' +
                    '</tr></thead>'));
                const table_body = $('<tbody></tbody>');
                for (let i = 0; i < clusters.length; i++) {
                    const cluster = clusters[i];
                    list_view.append(createArticleClusterView(cluster))
                }
                list_view.append(table_body);
            }
        });
        container.append($('<div class="table-responsive"></div>').append(list_view));
        container.append(pagination);
    }

    // Create a view to display a keyword cluster
    function createArticleClusterView(article_cluster){
        const cluster_no = article_cluster['Cluster'];
        const doc_ids = article_cluster['DocIds'];
        const color = color_plates[cluster_no-1];
        const article_cluster_view = $('<tr></tr>');
        const score = parseFloat(article_cluster['Score']).toFixed(2);
        // Create a button to show the keyword cluster
        const btn = $('<button type="button" class="btn btn-link" style="color:' + color+'">' +
            cluster_no + '</button>');
        btn.button();
        btn.click(function(event){
            // Highlight the dots of a specific keyword cluster
            const chart = new ScatterGraph(corpus_data, article_clusters, cluster_no);
        });
        article_cluster_view.append($('<td></td>').append(btn));
        if(score < 0.0){
            // Add a col to display score
            article_cluster_view.append($('<td>' + doc_ids.length +
                '<span class="text-danger">(' + score +  ')</span>' +
                '</td>'));
        }else{
            // Add a col to display score
            article_cluster_view.append($('<td>' + doc_ids.length +
                '<span>(' + score +  ')</span>' +
                '</td>'));
        }



        // Display TF-IDF terms
        const terms = article_cluster['Terms'].map(t => t['term']);

        const term_div = $('<td></td>');
        // // Display top 10 key phrases
        const text_span = $('<p></p>');
        text_span.text(terms.join(", "));
        term_div.append(text_span);
        article_cluster_view.append(term_div);

        return article_cluster_view;
    }


    console.log(article_clusters);
    function createUI(){
        $('#article_cluster_list').empty();
        const container = $('<div class="container-sm"></div>');
        // // Add header

        createPagination(container);

        $('#article_cluster_list').append(container);

    }

    createUI();

}
