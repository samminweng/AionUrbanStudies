// Create scatter graph
function ScatterGraph(_corpus_data, _cluster_data, _select_no) {
    const width = 600;
    const height = 600;
    const corpus_data = _corpus_data;
    const cluster_data = _cluster_data;


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

    // Get top N terms of a cluster by TF-IDF
    function get_cluster_terms(cluster_no, n) {
        // Cluster top 5 topics
        let cluster_terms = cluster_data.find(c => c['Cluster'] === cluster_no)['Terms'];
        cluster_terms.sort((a, b) => b['doc_ids'].length - a['doc_ids'].length);
        return cluster_terms;
    }

    // Convert the json data to Plotly js data format
    function convert_cluster_data_to_data_points() {
        let traces = [];
        // Convert the clustered data into the format for Plotly js chart
        for (const cluster of _cluster_data) {
            const cluster_no = cluster['Cluster'];
            const cluster_docs = corpus_data.filter(d => d['Cluster'] === cluster_no);
            const cluster_name = "" + cluster_no;
            let data_point = {'x': [], 'y': [], 'label': []};
            for (const doc of cluster_docs) {
                data_point['x'].push(doc.x);
                data_point['y'].push(doc.y);
                const terms = get_cluster_terms(cluster_no, 5);
                const term_text = terms.map(t => '<i>' + t['term'] + '</i>').join("<br>");
                // const percent = parseInt(100 * cluster_data.find(c => c['Cluster'] === cluster_no)['Percent']);
                // Tooltip label displays top 5 topics
                data_point['label'].push('<b>Cluster ' + cluster_name + '</b> has ' +  cluster_docs.length +
                    ' articles and score of ' + doc['Score'].toFixed(2) + '<br>' +
                    term_text);
            }
            // Trace setting
            let trace = {
                'x': data_point['x'], 'y': data_point['y'], 'text': data_point['label'],
                'name': cluster_name, 'mode': 'markers', 'type': 'scatter',
                'marker': {color: get_color(cluster), size: 5, line: {width: 0}},
                'hovertemplate': '%{text}'
            };
            // Update opacity based on the selection
            if (_select_no) {
                if (cluster_no === _select_no) {
                    trace['opacity'] = 1;
                } else {
                    trace['opacity'] = 0.05;
                }
            } else {
                trace['opacity'] = 1;
            }

            traces.push(trace);
        }
        // console.log(traces)
        return traces;
    }

    // Display top terms above the chart
    function display_top_terms(cluster_no) {
        $('#hover_info').empty();
        // const n = 10;
        const terms = get_cluster_terms(cluster_no, n);      // Get top 10 cluster topics
        const terms_text = terms.map(t => t['term']).join(", ");
        // Add the cluster heading
        const cluster_name = 'Article Cluster ' + cluster_no;
        $('#hover_info').append($('<div class="h5">' + cluster_name + '</div>'));
        $('#hover_info').append($('<div>' + terms_text + '</div>'));
        $('#hover_info').focus();
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
                showticklabels: false
            },
            yaxis: {
                showgrid: false,
                showline: false,
                zeroline: false,
                showticklabels: false
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
            hovermode: 'closest',
            config: {responsive: true}
        };
        const config = {
            displayModeBar: false // Hide the floating bar
        }
        // Get the cluster number
        Plotly.newPlot('cluster_chart', data_points, layout, config);
        // Define the chart on click and hover events

        const cluster_chart = document.getElementById('cluster_chart');


        // Add chart hover event
        cluster_chart.on('plotly_hover', function (data) {
            if (data.points.length > 0) {
                const point = data.points[0];
                if (point.data.name.includes('#')) {
                    const cluster_no = parseInt(point.data.name.split("#")[1]);
                    display_top_terms(cluster_no);
                }
            }
        }).on('plotly_unhover', function (data) {
            $('#hover_info').empty();
        });


    }


    // Create the network graph using D3 library
    function _createUI() {
        $('#cluster_chart').empty();
        $('#cluster_chart').css('width', width).css('height', height);
        drawChart();
        if (_select_no) {
            const cluster = cluster_data.find(c => c['Cluster'] === _select_no);
            const cluster_doc_list = new ClusterDocList(_select_no, corpus_data, cluster_data, get_color(cluster));
        } else {
            // Clean the right panel
            $('#cluster_doc_heading').empty();
            $('#cluster_terms').empty();
            $('#doc_list_heading').empty();
            $('#doc_list').empty();
        }
    }

    _createUI();
}

