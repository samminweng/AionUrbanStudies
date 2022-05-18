// Create scatter graph
function ScatterGraph(corpus_data, cluster_data, article_cluster_no, keyword_cluster_no) {
    const width = 600;
    const height = 700;
    const article_cluster = cluster_data.find(c => c['Cluster'] === article_cluster_no);
    const keyword_clusters = article_cluster['KeywordClusters'];
    let x_range = [0, 12];
    let y_range = [0, 12];
    // D3 category color pallets
    const color_plates = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22", "#17becf"];

    // Convert the keyword clusters to Plotly js data format
    function convert_keyword_clusters_to_data_points() {
        let traces = [];
        // Convert the clustered data into the format for Plotly js chart
        for (const keyword_cluster of keyword_clusters) {
            const group_no = keyword_cluster['Group'];
            const keywords = keyword_cluster['Key-phrases'];
            const x_pos = keyword_cluster['x'];
            const y_pos = keyword_cluster['y'];
            const score = keyword_cluster['score'];
            // Get the docs about keyword cluster
            // Each keyword is a dot
            let data_point = {'x': [], 'y': [], 'label': []};
            for (let i = 0; i < keywords.length; i++) {
                const keyword = keywords[i];
                data_point['x'].push(x_pos[i]);
                data_point['y'].push(y_pos[i]);
                // Tooltip label displays top 5 topics
                data_point['label'].push(
                    '<b>Keyword Cluster ' + group_no + '</b> (' + keywords.length +
                    ' keywords, ' + score.toFixed(2) + ' score)<br><extra></extra>' +
                    '<b>Keyword:</b> <i>' + keyword + '</i>');
            }
            // // Trace setting
            let trace = {
                'x': data_point['x'], 'y': data_point['y'], 'text': data_point['label'],
                'name': group_no, 'mode': 'markers', 'type': 'scatter',
                'marker': {color: color_plates[group_no - 1], size: 5, line: {width: 0}},
                'hovertemplate': '%{text}'
            };
            // Update opacity based on the selection
            if (keyword_cluster_no) {
                if (keyword_cluster_no === group_no) {
                    trace['opacity'] = 1;
                } else {
                    trace['opacity'] = 0.2;
                }
            } else {
                trace['opacity'] = 1;
            }


            traces.push(trace);

            const x_min = Math.floor(Math.min(...x_pos));
            // Update the range of x axis
            if (x_min < x_range[0]) {
                x_range[0] = x_min;
            }
            const x_max = Math.ceil(Math.max(...x_pos));
            if (x_max > x_range[1]) {
                x_range[1] = x_max;
            }
            const y_min = Math.floor(Math.min(...y_pos));
            if (y_min < y_range[0]) {
                y_range[0] = y_min;
            }
            const y_max = Math.ceil(Math.max(...y_pos));
            if (y_max > y_range[1]) {
                y_range[1] = y_max;
            }
        }
        return traces;
    }

    // Draw google chart
    function drawChart() {
        const data_points = convert_keyword_clusters_to_data_points();
        // Define the layout
        let layout = {
            // autosize: true,
            width: width,
            height: height,
            xaxis: {
                range:x_range,
                // showgrid: false,
                // showline: false,
                // zeroline: false,
                // showticklabels: false
            },
            yaxis: {
                range: y_range,
                // showgrid: false,
                // showline: false,
                // zeroline: false,
                // showticklabels: false
            },
            // Set the graph margin
            margin: {
                l: 20,
                r: 20,
                b: 20,
                t: 40
            },
            // Plot the legend outside the
            showlegend: true,
            // Display the legend horizontally
            legend: {
                "orientation": "v",
                font: {
                    size: 12,
                },
            },
            annotations: [],
            hovermode: 'closest',
            hoverlabel: {
                font: 18
            },
            config: {responsive: true}
        };

        const config = {
            displayModeBar: true, // Hide the floating bar
            modeBarButtonsToRemove: ['pan2d','select2d','lasso2d', 'zoom2d','zoomIn2d','zoomOut2d', 'toImage']
        }
        // Get the cluster number
        Plotly.newPlot('cluster_chart', data_points, layout, config);


    }

    // Create the network graph using D3 library
    function _createUI() {
        $('#cluster_chart').empty();
        $('#cluster_chart').css('width', width).css('height', height);
        drawChart();

        // Display all keyword clusters
        const view = new KeywordClusterList(corpus_data, cluster_data, article_cluster_no);
    }

    _createUI();
}

