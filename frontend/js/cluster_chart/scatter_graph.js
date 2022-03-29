// Create scatter graph
function ScatterGraph(_corpus_data, _cluster_data) {
    const width = 600;
    const height = 800;
    const corpus_data = _corpus_data;
    const cluster_data = _cluster_data;
    // Optimal color pallets for 31 colors from https://medialab.github.io/iwanthue/
    const color_plates = ["#45bf78", "#9857ca", "#57c356", "#dc67c8", "#87bb37", "#5970d8", "#beb43a",
        "#ad3488", "#508e2d", "#dc498a", "#68c194", "#dc385a", "#46c7ca", "#d5532c",
        "#609ad5", "#db9435", "#665fa2", "#96892e", "#be8fd8", "#478b4e", "#985489",
        "#a8b56d", "#a1475d", "#31947c", "#c95758", "#2a6a45", "#e387a5", "#656c29",
        "#e59671", "#9f4f2c", "#a07138"];

    // Get top N terms of a cluster by TF-IDF
    function get_cluster_terms(cluster_no, n) {
        // Cluster top 5 topics
        let cluster_terms = cluster_data.find(c => c['Cluster'] === cluster_no)['Terms'].slice(0, 10);
        cluster_terms.sort((a, b) => b['doc_ids'].length - a['doc_ids'].length);
        return cluster_terms.slice(0, n);
    }

    // Convert the json data to Plotly js data format
    function convert_cluster_data_to_data_points() {
        // Get the categorical color for each cluster
        const colors = function (cluster_no) {
            return (cluster_no < 0) ? "gray" : color_plates[cluster_no];
        };
        // Determine the opacity based on the outlier or cluster
        const opacity = function (cluster_no) {
            return (cluster_no < 0) ? 0.2 : 1.0;
        };
        let traces = [];

        // Convert the clustered data into the format for Plotly js chart
        for (const cluster of _cluster_data) {
            const cluster_no = cluster['Cluster'];
            const cluster_docs = corpus_data.filter(d => d['Cluster'] === cluster_no);
            const cluster_name = " #" + cluster_no;
            let data_point = {'x': [], 'y': [], 'label': []};
            for (const doc of cluster_docs) {
                data_point['x'].push(doc.x);
                data_point['y'].push(doc.y);
                const terms = get_cluster_terms(cluster_no, 5);
                const term_text = terms.map(t => t['term']).join("<br>");
                const percent = parseInt(100 * cluster_data.find(c => c['Cluster'] === cluster_no)['Percent']);
                // Tooltip label displays top 5 topics
                data_point['label'].push('<b>' + cluster_name + '</b> has ' + cluster_docs.length +
                    ' papers (' + percent + '%) <br>' +
                    'Silhouette score = <b>' + doc['Score'].toFixed(2) + '</b><br>'
                    + term_text);
            }
            // Trace setting
            let trace = {
                'x': data_point['x'], 'y': data_point['y'], 'text': data_point['label'],
                'name': cluster_name, 'mode': 'markers', 'type': 'scatter',
                'marker': {color: colors(cluster_no)}, opacity: opacity(cluster_no),
                'hovertemplate': '%{text}'
            };
            traces.push(trace);
        }

        // Sort traces by name
        traces.sort((a, b) => {
            if (a['name'].localeCompare(b['name']) !== 0) {
                const a_g = parseInt(a['name'].split("#")[1]);
                const b_g = parseInt(b['name'].split("#")[1]);
                if (a_g > b_g) {
                    return 1;
                } else {
                    return -1;
                }
            }
            return 0;
        });
        console.log(traces)
        return traces;
    }

    // Display top terms above the chart
    function display_top_terms(cluster_no) {
        $('#hover_info').empty();
        const n = 10;
        const terms = get_cluster_terms(cluster_no, n);      // Get top 10 cluster topics
        const terms_text = terms.map(t => t['term']).join(", ");
        // Add the cluster heading
        const cluster_name = 'Article Cluster #' + (cluster_no + 1);
        $('#hover_info').append($('<div class="h5">' + cluster_name + ' Top ' + n + ' Distinct Terms</div>'));
        $('#hover_info').append($('<div>' + terms_text + '</div>'));
        $('#hover_info').focus();
    }

    // Draw google chart
    function drawChart() {
        const data_points = convert_cluster_data_to_data_points();
        // Define the layout
        const layout = {
            autosize: true,
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
                    size: 10,
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
        const cluster_chart = document.getElementById('cluster_chart');
        // // Add chart onclick to toggle annotation
        cluster_chart.on('plotly_click', function (data) {
            const point = data.points[0];
            // Get the doc id from text
            let cluster_no = -1;
            if (point.data.name.includes('#')) {
                cluster_no = parseInt(point.data.name.split("#")[1]) - 1; // Group number -1
            }
            const cluster_doc_list = new ClusterDocList(cluster_no, corpus_data, cluster_data);
            // Add an annotation to the clustered dots
            const new_annotation = {
                x: point.xaxis.d2l(point.x),
                y: point.yaxis.d2l(point.y),
                bordercolor: point.fullData.marker.color,
                text: '<b>' + point.data.name + '</b>'
            };
            // Add onclick event to show/hide annotation of the cluster.
            const div = document.getElementById('cluster_chart');
            const newIndex = (div.layout.annotations || []).length;
            if (newIndex > 0) {
                // Find if any annotation of the cluster appears before.
                // If so, remove the annotation.
                div.layout.annotations.forEach((ann, index) => {
                    if (ann.text === new_annotation.text) {
                        Plotly.relayout('cluster_chart', 'annotations[' + index + ']', 'remove');
                        return;
                    }
                });
            }
            Plotly.relayout('cluster_chart', 'annotations[' + newIndex + ']', new_annotation);
        });
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
    }

    _createUI();
}

