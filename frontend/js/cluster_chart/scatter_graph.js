// Create scatter graph
function ScatterGraph(is_hide, _corpus_data, _cluster_data) {
    const width = 600;
    const height = 600;
    const corpus_data = _corpus_data;
    const cluster_data = _cluster_data;
    const initial_cluster = (is_hide) ? 0 : -1;
    // Optimal color pallets for 30 colors from http://vrl.cs.brown.edu/color
    // citation:
    // @article{gramazio-2017-ccd,
    //   author={Gramazio, Connor C. and Laidlaw, David H. and Schloss, Karen B.},
    //   journal={IEEE Transactions on Visualization and Computer Graphics},
    //   title={Colorgorical: creating discriminable and preferable color palettes for information visualization},
    //   year={2017}
    // }
    const color_plates = ["#41bbc5", "#256676", "#8de990", "#1c5f1e", "#4ca346", "#bfcd8e", "#754819", "#ea8244",
        "#8c1132", "#ea7c97", "#f4327e", "#d4afb9"];
    // Find the maximal cluster number as total number of clusters
    const total_clusters = corpus_data.map(c => c['Cluster']).reduce((p_value, c_value) => {
        return (p_value >= c_value) ? p_value : c_value;
    }, 0) + 1;
    console.log(total_clusters);

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
        let data_points = [];

        // Convert the clustered data into the format for Plotly js chart
        for (let cluster_no = initial_cluster; cluster_no < total_clusters; cluster_no++) {
            const cluster_docs = corpus_data.filter(d => d['Cluster'] === cluster_no);
            const group_name = "Group #" + (cluster_no + 1);
            if (cluster_docs.length > 0) {
                let data_point = {'x': [], 'y': [], 'label': []};
                for (const doc of cluster_docs) {
                    data_point['x'].push(doc.x);
                    data_point['y'].push(doc.y);
                    const terms = get_cluster_terms(cluster_no, 5);
                    const term_text = terms.map(t => t['term']).join("<br>");
                    const percent = parseInt(100 * cluster_data.find(c => c['Cluster'] === cluster_no)['Percent']);
                    // Tooltip label displays top 5 topics
                    data_point['label'].push('<b>' + group_name +'</b> has ' + cluster_docs.length +
                        ' papers (' + percent + '%)<br>' + term_text);
                }
                // const trace_name = 'Group #' + (cluster_no + 1);
                // Trace setting
                let trace = {
                    'x': data_point['x'], 'y': data_point['y'], 'text': data_point['label'],
                    'name': group_name, 'mode': 'markers', 'type': 'scatter',
                    'marker': {color: colors(cluster_no)}, opacity: opacity(cluster_no),
                    'hovertemplate': '%{text}'
                };
                data_points.push(trace);
            }
        }
        return data_points;
    }

    // Display top terms above the chart
    function display_top_terms(cluster_no) {
        $('#hover_info').empty();
        const n = 10;
        const terms = get_cluster_terms(cluster_no, n);      // Get top 10 cluster topics
        const terms_text = terms.map(t => t['term']).join(", ");
        // Add the cluster heading
        if (cluster_no !== -1) {
            $('#hover_info').append($('<div class="h5">Group #' + cluster_no + ' Top ' + n + ' Distinct Terms</div>'));
        } else {
            $('#hover_info').append($('<div class="h5">Outlier Top ' + n + ' Distinct terms</div>'));
        }
        $('#hover_info').append($('<div>' + terms_text + '</div>'));
        $('#hover_info').focus();
    }

    // Draw google chart
    function drawChart() {
        const data_points = convert_cluster_data_to_data_points();
        // Define the layout
        const options = {
            autosize: true,
            width: width,
            height: height,
            // Set the graph margin
            margin: {
                l: 10,
                r: 0,
                b: 20,
                t: 10
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
        Plotly.newPlot('cluster_chart', data_points, options, config);
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
                let cluster_no = -1;
                if (point.data.name.includes('#')) {
                    cluster_no = parseInt(point.data.name.split("#")[1]);
                }
                display_top_terms(cluster_no);
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

