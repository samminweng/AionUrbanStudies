// Create scatter graph
function ScatterGraph(corpus_data, cluster_data, _select_no) {
    const width = 600;
    const height = 600;
    // Collect all the cluster groups
    const cluster_groups = cluster_data.reduce((pre, cur) => {
        const _group_no = cur['cluster_group']
        if(!pre.includes(_group_no)){
            pre.push(_group_no);
        }
        return pre;
    }, []);

    // Convert the json data to Plotly js data format
    function convert_cluster_data_to_data_points() {
        let traces = [];
        for(const group_no of cluster_groups){
            const clusters = cluster_data.filter(c => c['cluster_group'] === group_no);
            const doc_count = clusters.reduce((pre, cur) => pre + cur['doc_ids'].length, 0);
            // Convert the clustered data into the format for Plotly js chart
            for (const cluster of clusters) {
                const cluster_no = cluster['cluster'];
                const cluster_docs = corpus_data.filter(d => d['Cluster'] === cluster_no);
                // const cluster_name = "" + cluster_no;
                let data_point = {'x': [], 'y': [], 'label': []};
                for (const doc of cluster_docs) {
                    data_point['x'].push(doc.x);
                    data_point['y'].push(doc.y);
                    data_point['label'].push(
                        '<b>' + clusters.length + ' abstract clusters</b> in the region contain ' + doc_count + ' abstracts.'
                    );
                }
                // Trace setting
                let trace = {
                    'x': data_point['x'], 'y': data_point['y'], 'text': data_point['label'],
                    'name': cluster_no,
                    'mode': 'markers', 'type': 'scatter',
                    'marker': {color: get_color(cluster), size: 5, line: {width: 0}},
                    'hovertemplate': '%{text}<extra></extra>', 'opacity': 1
                };
                // Update opacity based on the selection
                if (_select_no){
                    if(cluster_no === _select_no) {
                        trace['opacity'] = 1;
                    }else{
                        trace['opacity'] = 0.7;
                    }
                }
                traces.push(trace);
            }
        }


        // console.log(traces)
        return traces;
    }

    // Draw google chart
    function drawChart() {
        const data_points = convert_cluster_data_to_data_points();
        // Define the layout
        const layout = {
            // autosize: true,
            width: width,
            height: height,
            xaxis: {
                showgrid: false,
                showline: false,
                zeroline: false,
                showticklabels: false,
                fixedrange: true
            },
            yaxis: {
                showgrid: false,
                showline: false,
                zeroline: false,
                showticklabels: false,
                fixedrange: true
            },
            // Set the graph margin
            margin: {
                l: 0,
                r: 0,
                b: 0,
                t: 0
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
            hovermode: 'closest'
        };
        const config = {
            responsive: true,
            displayModeBar: false // Hide the floating bar
        }
        // Get the cluster number
        Plotly.newPlot('cluster_chart', data_points, layout, config);
        // Define the chart on click and hover events
        const chart = document.getElementById('cluster_chart');
        // Add chart click event
        chart.on('plotly_click', function(data){
            if (data.points.length > 0){
                const point = data.points[0];
                const cluster_no = parseInt(point.data.name);
                if(cluster_no){
                    const selected_cluster = cluster_data.find(c => c['cluster'] === cluster_no);
                    const grouped_clusters = cluster_data.filter(c => c['cluster_group'] === selected_cluster['cluster_group']);
                    // Update the opacity
                    Plotly.restyle(chart, {opacity: 0.7}, cluster_data.map(c => c['cluster']-1));
                    Plotly.restyle(chart, {opacity: 1}, grouped_clusters.map(c => c['cluster']-1));
                    const list = new AbstractClusterList(corpus_data, cluster_data, grouped_clusters);
                }
            }
        });

    }


    // Create the network graph using D3 library
    function _createUI() {
        $('#cluster_chart').empty();
        $('#cluster_chart').css('width', width).css('height', height);
        drawChart();
    }

    _createUI();
}

