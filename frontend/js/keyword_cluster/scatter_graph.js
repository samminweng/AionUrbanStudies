// Create scatter graph
function ScatterGraph(corpus_data, cluster_data, select_no) {
    const width = 600;
    const height = 700;
    const article_cluster = cluster_data.find(c => c['Cluster'] === select_no);
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
                'name': 'Keyword Cluster ' + group_no, 'mode': 'markers', 'type': 'scatter',
                'marker': {color: color_plates[group_no - 1], size: 5, line: {width: 0}},
                'hovertemplate': '%{text}'
            };
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

    // Display top terms above the chart
    function display_top_terms(cluster_no) {
        $('#hover_info').empty();
        const n = 10;
        // const terms = get_cluster_terms(cluster_no, n);      // Get top 10 cluster topics
        // const terms_text = terms.map(t => t['term']).join(", ");
        // Add the cluster heading
        const cluster_name = 'Article Cluster ' + cluster_no;
        $('#hover_info').append($('<div class="h5">' + cluster_name + '</div>'));
        // $('#hover_info').append($('<div>' + terms_text + '</div>'));
        $('#hover_info').focus();
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
        // Define the chart on click and hover events

        const cluster_chart = document.getElementById('cluster_chart');
        // // Add chart onclick to toggle annotation
        cluster_chart.on('plotly_click', function (data) {
            const point = data.points[0];
            const group_no = parseInt(point.data.name.split("Keyword Cluster")[1]);
            const keyword_cluster = keyword_clusters.find(c => c['Group'] === group_no);
            const docs = corpus_data.filter(d => keyword_cluster['DocIds'].includes(d['DocId']));
            const color = color_plates[group_no-1];
            const keyword_view = new KeywordView(keyword_cluster, docs, color);
            // // // Add an annotation to the clustered dots
            // const new_annotation = {
            //     x: point.xaxis.d2l(point.x),
            //     y: point.yaxis.d2l(point.y),
            //     bordercolor: point.fullData.marker.color,
            //     text: '<b>Article Cluster #' + cluster_no + '</b>'
            // };
            // // Add onclick event to show/hide annotation of the cluster.
            // const div = document.getElementById('cluster_chart');
            // const newIndex = (div.layout.annotations || []).length;
            // if (newIndex > 0) {
            //     // Find if any annotation of the cluster appears before.
            //     // If so, remove the annotation.
            //     div.layout.annotations.forEach((ann, index) => {
            //         if (ann.text === new_annotation.text) {
            //             Plotly.relayout('cluster_chart', 'annotations[' + index + ']', 'remove');
            //             return;
            //         }
            //     });
            // }
            // Plotly.relayout('cluster_chart', 'annotations[' + newIndex + ']', new_annotation);
        });
        //
        // // Add chart hover event
        // cluster_chart.on('plotly_hover', function (data) {
        //     if (data.points.length > 0) {
        //         const point = data.points[0];
        //         if (point.data.name.includes('#')) {
        //             const cluster_no = parseInt(point.data.name.split("#")[1]);
        //             display_top_terms(cluster_no);
        //         }
        //     }
        // }).on('plotly_unhover', function (data) {
        //     $('#hover_info').empty();
        // });


    }

    // Create the network graph using D3 library
    function _createUI() {
        $('#cluster_chart').empty();
        $('#cluster_chart').css('width', width).css('height', height);
        drawChart();
        // if (_select_no) {
        //     const cluster_doc_list = new ClusterDocList(_select_no, corpus_data, cluster_data);
        // } else {
        //     // Clean the right panel
        //     $('#cluster_doc_heading').empty();
        //     $('#cluster_terms').empty();
        //     $('#doc_list_heading').empty();
        //     $('#doc_list').empty();
        // }
    }

    _createUI();
}

