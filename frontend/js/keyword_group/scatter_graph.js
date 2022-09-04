// Create scatter graph
function ScatterGraph(corpus_data, cluster_data, cluster_no, keyword_group_no) {
    const width = 600;
    const height = 600;
    const abstract_cluster = cluster_data.find(c => c['cluster'] === cluster_no);
    const keyword_groups = abstract_cluster['keyword_groups'];
    let x_range = [0, 12];
    let y_range = [0, 12];

    // Get small rectangles of all dots
    // Ref: https://www.geeksforgeeks.org/coordinates-rectangle-given-points-lie-inside/?ref=rp
    function get_small_rectangles(x_arr, y_arr){
        // find max and min of x position
        const x_max = x_arr.reduce((a,b) => Math.max(a,b));
        const x_min = x_arr.reduce((a,b) => Math.min(a,b));
        const x_center = (x_max + x_min)/2;
        // Update x_range
        x_range[0] = Math.min(x_center - 2, x_min);
        x_range[1] = Math.max(x_center + 2, x_max);
        // find max and min of y position
        const y_max =  y_arr.reduce((a,b) => Math.max(a,b));
        const y_min = y_arr.reduce((a,b) => Math.min(a,b));
        const y_center = (y_max + y_min)/2;
        y_range[0] = Math.min(y_center - 2, y_min);
        y_range[1] = Math.max(y_center + 2, y_max);
    }

    // Convert the keyword clusters to Plotly js data format
    function convert_keyword_clusters_to_data_points() {
        let traces = [];
        let x_arr = [], y_arr = [];
        // Convert the clustered data into the format for Plotly js chart
        for (const keyword_group of keyword_groups) {
            if(keyword_group['score'] >0 ){
                const group_no = keyword_group['group'];
                const keywords = keyword_group['keywords'];
                const x_pos = keyword_group['x'];
                const y_pos = keyword_group['y'];
                const score = keyword_group['score'];
                // Get the docs about keyword cluster
                // Each keyword is a dot
                let data_point = {'x': [], 'y': [], 'label': []};
                for (let i = 0; i < keywords.length; i++) {
                    const keyword = keywords[i];
                    data_point['x'].push(x_pos[i]);
                    data_point['y'].push(y_pos[i]);
                    // Tooltip label displays top 5 topics
                    data_point['label'].push(
                        '<b>Keyword Group ' + group_no + '</b> (' + keywords.length +
                        ' keywords, ' + score.toFixed(2) + ' score)<br><extra></extra>');
                }
                // // Trace setting
                let trace = {
                    'x': data_point['x'], 'y': data_point['y'], 'text': data_point['label'],
                    'name': group_no, 'mode': 'markers', 'type': 'scatter',
                    'marker': {color: color_plates[group_no - 1], size: 5, line: {width: 0}},
                    'hovertemplate': '%{text}', 'opacity': 1
                };
                // Update opacity based on the selection
                if (keyword_group_no) {
                    if (keyword_group_no === group_no) {
                        trace['opacity'] = 1;
                    } else {
                        trace['opacity'] = 0.5;
                    }
                }

                traces.push(trace);
                x_arr = x_arr.concat(x_pos);
                y_arr = y_arr.concat(y_pos);
            }



        }
        get_small_rectangles(x_arr, y_arr);
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
                showgrid: true,
                showline: false,
                zeroline: false,
                showticklabels: false
            },
            yaxis: {
                range: y_range,
                showgrid: true,
                showline: false,
                zeroline: false,
                showticklabels: false
            },
            // Set the graph margin
            margin: {
                l: 20,
                r: 20,
                b: 20,
                t: 20
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
            displayModeBar: false, // Hide the floating bar
        }
        // Get the cluster number
        Plotly.newPlot('cluster_chart', data_points, layout, config);
        // Define the chart on click and hover events
        const chart = document.getElementById('cluster_chart');
        // Add chart click event
        chart.on('plotly_click', function(data){
            if (data.points.length > 0){
                const point = data.points[0];
                const group_no = parseInt(point.data.name);
                if(group_no){
                    const keyword_group = keyword_groups.find(c => c['group'] === group_no);
                    const other_groups = keyword_groups.filter(c => c['group'] !== group_no);
                    // Update the opacity
                    // Plotly.restyle(chart, {opacity: 0.2}, other_groups.map(c => c['group']-1));
                    // Plotly.restyle(chart, {opacity: 1}, [group_no-1]);
                    // Display keyword cluster view
                    const docs = corpus_data.filter(d => keyword_group['doc_ids'].includes(d['DocId']));
                    const view = new KeywordClusterView(keyword_group, docs);
                }
            }
        });

    }

    // Create the network graph using D3 library
    function _createUI() {
        $('#cluster_chart').empty();
        $('#cluster_chart').css('width', width).css('height', height);
        drawChart();
        // Display all keyword clusters
        const view = new KeywordGroupList(corpus_data, cluster_data, cluster_no);

        $('#keyword_group_view').empty();
        $('#doc_list_view').empty();
    }

    _createUI();
}

