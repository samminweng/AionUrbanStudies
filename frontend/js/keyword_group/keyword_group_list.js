// Create a view to displays all keyword clusters
function KeywordClusterList(corpus_data, cluster_data, article_cluster_no){
    const article_cluster = cluster_data.find(c => c['Cluster'] === article_cluster_no);
    const keyword_groups = article_cluster['KeywordGroups'].filter(group => group['score'] >0);
    const cluster_docs = corpus_data.filter(d => article_cluster['DocIds'].includes(d['DocId']));

    // Create a pagination to show the article clusters
    function createPagination() {
        const container = $('<div></div>');
        // Create the pagination
        const pagination = $('<div></div>');
        // Create the list
        const table = $('<table class="table table-sm small p-0"></table>');
        // Pagination
        pagination.pagination({
            dataSource: function (done) {
                let result = [];
                for (let i = 0; i < keyword_groups.length; i++) {
                    result.push(keyword_groups[i]);
                }
                done(result);
            },
            totalNumber: keyword_groups.length,
            pageSize: 5,
            showNavigator: true,
            formatNavigator: '<span style="color: #f00"><%= currentPage %></span>/<%= totalPage %> pages, ' +
                '<%= totalNumber %> keyword clusters',
            callback: function (clusters, pagination) {
                table.empty();
                // Add each keyword cluster
                table.append($('<thead><tr>' +
                               '<th>Keyword Group (Score)</th>' +
                               // '<th>Silhouette Score</th>' +
                               '<th>Keywords</th></tr></thead>'));
                const table_body = $('<tbody></tbody>');
                for (let i = 0; i < clusters.length; i++) {
                    const cluster = clusters[i];
                    table_body.append(createKeywordGroup(cluster))
                }
                table.append(table_body);
            }
        });
        container.append($('<div class="table-responsive-sm"></div>').append(table));
        if(keyword_groups.length > 5){
            container.append(pagination);
        }
        return container;
    }


    // Create a view to display a keyword cluster
    function createKeywordGroup(keyword_group){
        const group_no = keyword_group['Group'];
        const color = (parseFloat(keyword_group['score']) > 0 ? color_plates[group_no-1] : 'gray');
        const keyword_cluster_view = $('<tr></tr>');
        const score = parseFloat(keyword_group['score']).toFixed(2);
        // Create a button to show the keyword cluster
        const btn = $('<button type="button" class="btn btn-link btn-sm" style="color:' + color+'">' +
                        keyword_group['Group']+ ' <span style="color:' + color+'">(' + score +')</span></button>');
        btn.button();
        btn.click(function(event){
            // Highlight the dots of a specific keyword cluster
            const graph = new ScatterGraph(corpus_data, cluster_data, article_cluster_no, group_no);
            // Display the keyword cluster view
            const keyword_cluster = keyword_groups[group_no-1];
            const docs = cluster_docs.filter(d => keyword_cluster['DocIds'].includes(d['DocId']));
            const view = new KeywordClusterView(keyword_groups[group_no-1], docs, color_plates[group_no-1]);
        });
        // Add a col to display
        keyword_cluster_view.append($('<td></td>').append(btn));

        // Display key phrases
        const keywords = keyword_group['Key-phrases'].sort((a, b) => a.localeCompare(b));
        // Display top 10 key phrases
        const keyword_div = $('<div class="container-sm"></div>');
        const max_size = Math.min(10, keywords.length);
        const sample_keywords = keywords.slice(0, max_size);
        let row;
        for(let i =0; i< max_size; i++){
            const keyword = sample_keywords[i];
            if(i %1 === 0){
                row = $('<div class="row"></div>');
                keyword_div.append(row);
            }
            row.append($('<div class="col text-truncate text-start">' + keyword + '</div>'));
        }
        keyword_cluster_view.append($('<td></td>').append(keyword_div));
        // Long list of key phrases
        if(keywords.length > max_size){
            // Create a more btn to view more topics
            const more_btn = $('<span class="text-muted text-end">MORE (' + keywords.length + ') ' +
                '<span class="ui-icon ui-icon-plus"></span></span>');
            // Create a few btn
            const less_btn = $('<div class="btn btn-sm text-muted">LESS<span class="ui-icon ui-icon-minus"></span></div>');
            // Display more key phrases
            more_btn.click(function(event){
                keyword_div.find(".text-truncate").remove();
                for(const keyword of keywords){
                    keyword_div.prepend($('<div class="col text-truncate text-start">' + keyword + '</div>'));
                }
                // Display 'less' btn only
                more_btn.hide();
                less_btn.show();
            });
            // Display top five key phrases
            less_btn.click(function(event){
                keyword_div.find(".text-truncate").remove();
                let row;
                for(let i =0; i< max_size; i++){
                    const keyword = sample_keywords[i];
                    if(i %1 === 0){
                        row = $('<div class="row"></div>');
                        keyword_div.prepend(row);
                    }
                    row.append($('<div class="col text-truncate text-start">' + keyword + '</div>'));
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
        $('#keyword_group_list').empty();
        const container = $('<div class="container-sm small"></div>');
        // Add header
        container.append($('<div class="row">Abstract Cluster #' + article_cluster_no + '  has ' +
                                                  keyword_groups.length + ' keyword groups</div>'));
        container.append(createPagination());
        $('#keyword_group_list').append(container);
    }

    createUI();
}
