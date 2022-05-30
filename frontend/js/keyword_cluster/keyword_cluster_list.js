// Create a view to displays all keyword clusters
function KeywordClusterList(corpus_data, cluster_data, article_cluster_no){
    const article_cluster = cluster_data.find(c => c['Cluster'] === article_cluster_no);
    const keyword_clusters = article_cluster['KeywordClusters'];
    const cluster_docs = corpus_data.filter(d => article_cluster['DocIds'].includes(d['DocId']));

    // Create a pagination to show the article clusters
    function createPagination() {
        const container = $('<div></div>');
        // Create the pagination
        const pagination = $('<div></div>');
        // Create the list
        const table = $('<table class="table table-sm"></table>');
        // Pagination
        pagination.pagination({
            dataSource: function (done) {
                let result = [];
                for (let i = 0; i < keyword_clusters.length; i++) {
                    result.push(keyword_clusters[i]);
                }
                done(result);
            },
            totalNumber: keyword_clusters.length,
            pageSize: 5,
            showNavigator: true,
            formatNavigator: '<span style="color: #f00"><%= currentPage %></span>/<%= totalPage %> pages, ' +
                '<%= totalNumber %> keyword clusters',
            callback: function (clusters, pagination) {
                table.empty();
                // Add each keyword cluster
                table.append($('<thead><tr>' +
                               '<th>Keyword Cluster</th>' +
                               '<th>Keywords</th></tr></thead>'));
                const table_body = $('<tbody></tbody>');
                for (let i = 0; i < clusters.length; i++) {
                    const cluster = clusters[i];
                    table_body.append(createKeywordCluster(cluster))
                }
                table.append(table_body);
            }
        });
        container.append($('<div class="table-responsive-sm"></div>').append(table));
        if(keyword_clusters.length > 5){
            container.append(pagination);
        }
        return container;
    }


    // Create a view to display a keyword cluster
    function createKeywordCluster(keyword_cluster){
        const group_no = keyword_cluster['Group'];
        const color = color_plates[group_no-1];
        const keyword_cluster_view = $('<tr></tr>');
        const score = parseFloat(keyword_cluster['score']).toFixed(2);
        // Create a button to show the keyword cluster
        const btn = $('<button type="button" class="btn btn-link" style="color:' + color+'">' +
            keyword_cluster['Group']+ ' <span style="color:' + color+'">(' + score +')</span></button>');
        btn.button();
        btn.click(function(event){
            // Highlight the dots of a specific keyword cluster
            const graph = new ScatterGraph(corpus_data, cluster_data, article_cluster_no, group_no);
            // Display the keyword cluster view
            const keyword_cluster = keyword_clusters[group_no-1];
            const docs = cluster_docs.filter(d => keyword_cluster['DocIds'].includes(d['DocId']));
            const view = new KeywordClusterView(keyword_clusters[group_no-1], docs, color_plates[group_no-1]);
        });

        // Add a col to display
        keyword_cluster_view.append($('<td></td>').append(btn));
        // Display key phrases
        const keywords = keyword_cluster['Key-phrases'];
        // Display top 10 key phrases
        const keyword_div = $('<div class="container-sm small"></div>');
        const max_size = 6;
        for(const keyword of keywords.slice(0, max_size)){
            keyword_div.append($('<div class="btn btn-sm text-truncate text-start" style="width: 200px;">' + keyword + '</div>'));
        }
        keyword_cluster_view.append($('<td></td>').append(keyword_div));
        // Long list of key phrases
        if(keywords.length > max_size){
            // Create a more btn to view more topics
            const more_btn = $('<div class="btn btn-sm text-muted text-end">MORE (' + keywords.length + ') ' +
                '<span class="ui-icon ui-icon-plus"></span></div>');
            // Create a few btn
            const less_btn = $('<div class="btn btn-sm text-muted">LESS<span class="ui-icon ui-icon-minus"></span></div>');
            // Display more key phrases
            more_btn.click(function(event){
                keyword_div.find(".text-truncate").remove();
                for(const keyword of keywords){
                    keyword_div.prepend($('<div class="btn btn-sm text-truncate text-start" ' +
                        'style="width: 200px;">' + keyword + '</div>'));
                }
                // Display 'less' btn only
                more_btn.hide();
                less_btn.show();
            });
            // Display top five key phrases
            less_btn.click(function(event){
                keyword_div.find(".text-truncate").remove();
                for(const keyword of keywords.slice(0, max_size)){
                    keyword_div.prepend($('<div class="btn btn-sm text-truncate text-start" ' +
                        'style="width: 200px;">' + keyword + '</div>'));
                }
                more_btn.show();
                less_btn.hide();
            });

            // By default, display more btn only.
            more_btn.show();
            less_btn.hide();
            const btn_div = $('<div class="text-end"></div>');
            btn_div.append(more_btn);
            btn_div.append(less_btn);
            keyword_div.append(btn_div);
        }
        return keyword_cluster_view;
    }


    function createUI(){
        $('#keyword_cluster_list').empty();
        const container = $('<div class="container"></div>');
        // Add header
        container.append($('<div class="row mb-3">Article Cluster #' + article_cluster_no + '  has ' +
                                                  keyword_clusters.length + ' keyword clusters</div>'));

        container.append(createPagination());
        $('#keyword_cluster_list').append(container);
    }

    createUI();
}
