// Create scatter graph
function ScatterGraph(corpus_data, cluster_data, _select_no) {
    const width = 600;
    const height = 650;

    // Collect all the groups
    const group_data = cluster_data.reduce((pre, cur) => {
        const _group_no = cur['Group']
        if(!pre.includes(_group_no)){
            pre.push(_group_no);
        }
        return pre;
    }, []);
    // console.log(group_data);
    // Get the cluster color by group number
    function get_color(article_cluster){
        const cluster_no = article_cluster['Cluster'];
        const group_no = article_cluster['Group'];
        // Get the group colors < group_no
        let index = 0;
        for(let i=1; i < group_no; i++){
            index += group_color_plates[i].length;
        }
        let color_index = cluster_no - index - 1;
        return group_color_plates[group_no][color_index];
    }

    // Convert the json data to Plotly js data format
    function convert_cluster_data_to_data_points() {
        let traces = [];
        for(const group_no of group_data){
            const clusters = cluster_data.filter(c => c['Group'] === group_no);
            const doc_count = clusters.reduce((pre, cur) => pre + cur['DocIds'].length, 0);
            // Convert the clustered data into the format for Plotly js chart
            for (const cluster of clusters) {
                const cluster_no = cluster['Cluster'];
                const cluster_docs = corpus_data.filter(d => d['Cluster'] === cluster_no);
                // const cluster_name = "" + cluster_no;
                let data_point = {'x': [], 'y': [], 'label': []};
                for (const doc of cluster_docs) {
                    data_point['x'].push(doc.x);
                    data_point['y'].push(doc.y);
                    data_point['label'].push(
                        '<b>' + clusters.length + ' abstract clusters</b> in the region contain ' + doc_count + ' articles.'
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
                        trace['opacity'] = 0.2;
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
                    const selected_cluster = cluster_data.find(c => c['Cluster'] === cluster_no);
                    const grouped_clusters = cluster_data.filter(c => c['Group'] === selected_cluster['Group']);
                    // Update the opacity
                    Plotly.restyle(chart, {opacity: 0.5}, cluster_data.map(c => c['Cluster']-1));
                    Plotly.restyle(chart, {opacity: 1}, grouped_clusters.map(c => c['Cluster']-1));
                    const list = new ArticleClusterList(corpus_data, cluster_data, grouped_clusters);
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

