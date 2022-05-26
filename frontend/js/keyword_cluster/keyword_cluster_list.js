// Create a view to displays all keyword clusters
function KeywordClusterList(corpus_data, cluster_data, article_cluster_no){
    const article_cluster = cluster_data.find(c => c['Cluster'] === article_cluster_no);
    const keyword_clusters = article_cluster['KeywordClusters'];
    const cluster_docs = corpus_data.filter(d => article_cluster['DocIds'].includes(d['DocId']));

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
        const max_size = 8;
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
        // Add each keyword cluster
        const table = $('<table class="table table-sm">' +
            '<thead><tr>' +
            '<th>Keyword Cluster</th>' +
            '<th>Keywords</th></tr></thead></table>');
        const tbody = $('<tbody></tbody>');
        for(const keyword_cluster of keyword_clusters){
            tbody.append(createKeywordCluster(keyword_cluster));
        }
        table.append(tbody);
        container.append(table);
        $('#keyword_cluster_list').append(container);
    }

    createUI();
}
// const avg_score = keyword_clusters.map(c => parseFloat(c['score'].toFixed(2)))
//                                   .reduce((pre, cur) => pre + cur, 0.0)/keyword_clusters.length;
// const weight_avg_score = get_weighted_average_score(keyword_clusters);
// // Get weighted average score
// function get_weighted_average_score(keyword_clusters){
//     let results = keyword_clusters.map(c => {
//         const weight = c['Key-phrases'].length;
//         const sum = weight * c['score'];
//         return [sum, weight];
//     }).reduce((pre, cur) =>{
//         return [pre[0] + cur[0], pre[1] + cur[1]];
//     }, [0, 0]);
//     const weight_sum = results[0];
//     const weight_total = results[1];
//     return weight_sum/weight_total;
// }
