// Create a view to displays all keyword clusters
function KeywordClusterList(corpus_data, cluster_data, article_cluster_no){
    const article_cluster = cluster_data.find(c => c['Cluster'] === article_cluster_no);
    const keyword_clusters = article_cluster['KeywordClusters'];
    const cluster_docs = corpus_data.filter(d => article_cluster['DocIds'].includes(d['DocId']));
    // D3 category color pallets
    const color_plates = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b", "#e377c2",
                          "#7f7f7f", "#bcbd22", "#17becf"];
    const avg_score = keyword_clusters.map(c => parseFloat(c['score'].toFixed(2)))
                                      .reduce((pre, cur) => pre + cur, 0.0)/keyword_clusters.length;
    const weight_avg_score = get_weighted_average_score(keyword_clusters);

    // Get weighted average score
    function get_weighted_average_score(keyword_clusters){
        let results = keyword_clusters.map(c => {
            const weight = c['Key-phrases'].length;
            const sum = weight * c['score'];
            return [sum, weight];
        }).reduce((pre, cur) =>{
            return [pre[0] + cur[0], pre[1] + cur[1]];
        }, [0, 0]);
        const weight_sum = results[0];
        const weight_total = results[1];
        return weight_sum/weight_total;
    }


    // Create a view to display a keyword cluster
    function createKeywordCluster(keyword_cluster){
        const group_no = keyword_cluster['Group'];
        const color = color_plates[group_no-1];
        const keyword_cluster_view = $('<div class="row"></div>');
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
        keyword_cluster_view.append($('<div class="col-sm-2"></div>').append(btn));
        // Display key phrases
        const keywords = keyword_cluster['Key-phrases'];
        const keyword_div = $('<div class="col-sm-10"></div>');
        // Display top 10 key phrases
        const text_span = $('<p></p>');
        text_span.text(keywords.slice(0, 8).join(", "));
        keyword_div.append(text_span);
        keyword_cluster_view.append(keyword_div);
        // Long list of key phrases
        if(keywords.length > 8){
            const btn_div = $('<div></div>');
            // Create a more btn to view more topics
            const more_btn = $('<span class="text-muted">MORE (' + keywords.length + ') ' +
                '<span class="ui-icon ui-icon-plus"></span></span>');
            // Create a few btn
            const less_btn = $('<span class="text-muted">LESS<span class="ui-icon ui-icon-minus"></span></span>');
            more_btn.css("font-size", "0.8em");
            less_btn.css("font-size", "0.8em");
            // Display more key phrases
            more_btn.click(function(event){
                text_span.text(keywords.join(", "));
                // Display 'less' btn only
                more_btn.hide();
                less_btn.show();
            });
            // Display top five key phrases
            less_btn.click(function(event){
                text_span.text(keywords.slice(0, 10).join(", "));
                more_btn.show();
                less_btn.hide();
            });

            // By default, display more btn only.
            more_btn.show();
            less_btn.hide();
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
        container.append($('<div class="row mb-3">Article Cluster #' + article_cluster_no +
            '  has ' + keyword_clusters.length + ' keyword clusters with weighted average = ' + weight_avg_score.toFixed(2) +
            '</div>'));
        // Add each keyword cluster
        container.append($('<div class="row">' +
            '<div class="col-sm-2 fw-bold">Keyword Cluster</div>' +
            '<div class="col-sm-10 fw-bold">Keywords</div></div>'))
        for(const keyword_cluster of keyword_clusters){
            container.append(createKeywordCluster(keyword_cluster));
        }
        $('#keyword_cluster_list').append(container);
        // Display
        const keyword_cluster = keyword_clusters[0];
        const docs = cluster_docs.filter(d => keyword_cluster['DocIds'].includes(d['DocId']));
        const view = new KeywordClusterView(keyword_clusters[0], docs);
    }


    createUI();

}
