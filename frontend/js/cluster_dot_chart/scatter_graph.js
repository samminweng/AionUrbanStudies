// Create scatter graph
function ScatterGraph(is_hide, approach, cluster_chart_data, cluster_topics, doc_data) {
    const width = 600;
    const height = 500;
    // Find the maximal cluster number as total number of clusters
    const total_clusters = cluster_chart_data.map(c => c[approach + '_Cluster']).reduce((p_value, c_value) => {
        return (p_value >= c_value) ? p_value : c_value;
    }, 0);
    const clusters = cluster_topics[approach];

    // Get the color of collocation
    const colors = function (cluster_no) {
        // Optimal color pallets for 23 clusters
        // ref: http://vrl.cs.brown.edu/color
        // const color_plates = ["rgb(104,175,252)", "rgb(79,40,175)", "rgb(153,109,219)", "rgb(47,66,133)", "rgb(62,234,239)",
        //     "rgb(37,115,139)", "rgb(179,228,103)", "rgb(39,122,53)", "rgb(103,240,89)", "rgb(117,72,25)",
        //     "rgb(252,206,106)", "rgb(179,65,108)", "rgb(196,145,150)", "rgb(192,0,24)", "rgb(254,133,173)",
        //     "rgb(248,35,135)", "rgb(254,143,6)", "rgb(169,190,175)", "rgb(178,139,40)", "rgb(239,102,240)",
        //     "#1e90ff", "#db7093", "#b0e0e6"];
        const color_plates = d3.schemeCategory10;
        return (cluster_no < 0) ? "gray" : color_plates[cluster_no];
    }
    // Determine the opacity based on the outlier or cluster
    const opacity = function (cluster_no) {
        return (cluster_no < 0) ? 0.2 : 1.0;
    }

    // Get top N topics of a cluster
    function get_cluster_topics(cluster_no, n) {
        // Cluster top 5 topics
        const topics = clusters.find(c => c['Cluster'] === cluster_no)['TopicN-gram'].slice(0, n);
        return topics;
    }


    // Convert the json data to Plotly js data format
    function convert_cluster_data_to_chart_format() {
        let data = [];
        const initial_cluster = (is_hide) ? 0 : -1;
        // Convert the clustered data into the format for Plotly js chart
        for (let cluster_no = initial_cluster; cluster_no <= total_clusters; cluster_no++) {
            const cluster_data = cluster_chart_data.filter(d => d[approach + '_Cluster'] === cluster_no);
            if (cluster_data.length > 0) {
                let data_point = {'x': [], 'y': [], 'label': []};
                for (const dot of cluster_data) {
                    data_point['x'].push(dot.x);
                    data_point['y'].push(dot.y);
                    const topics = get_cluster_topics(cluster_no, 5);
                    const topic_text = topics.map(t => t['topic']).join("<br>");
                    // Tooltip label displays top 5 topics
                    data_point['label'].push('<b>Cluster #' + cluster_no + '</b><br>' + topic_text);
                }
                // Trace setting
                let trace = {
                    'x': data_point['x'], 'y': data_point['y'], 'text': data_point['label'],
                    'name': 'Cluster #' + cluster_no, 'mode': 'markers', 'type': 'scatter',
                    'marker': {color: colors(cluster_no)}, opacity: opacity(cluster_no),
                    'hovertemplate': '%{text}'
                };
                data.push(trace);
            }
        }
        return data;
    }


    // Draw google chart
    function drawChart() {
        const data = convert_cluster_data_to_chart_format();
        // Define the layout
        const option = {
            autosize: false,
            width: width,
            height: height-20,
            // Set the graph margin
            margin: {
                l: 40,
                r: 10,
                b: 20,
                t: 10,
                pad: 3
            },
            // Plot the legend outside the
            showlegend: true,
            // // Display the legend horizontally
            legend: {
                "orientation": "h",
                font: {
                    size: 10,
                },
            },
            annotations: [],
            hovermode: 'closest',
            config: { responsive: true }
        }

        // Get the cluster number
        Plotly.newPlot('cluster_dot_chart', data, option);
        let is_lock = false;
        const cluster_dot_chart = document.getElementById('cluster_dot_chart');
        // Add chart onclick to toggle annotation
        cluster_dot_chart.on('plotly_click', function (data) {
            const point = data.points[0];
            // Get the doc id from text
            const cluster_no = parseInt(point.data.name.split("#")[1]);
            // Create a list of cluster doc
            const cluster = clusters.find(c => c['Cluster'] === cluster_no);
            const cluster_doc_list = new ClusterDocList(cluster, doc_data);

            // Add an annotation to the clustered dots.
            const new_annotation = {
                x: point.xaxis.d2l(point.x),
                y: point.yaxis.d2l(point.y),
                bordercolor: point.fullData.marker.color,
                text: '<b>Cluster ' + cluster_no + '</b>'
            };

            // Add onclick event to show/hide annotation of the cluster.
            const div = document.getElementById('cluster_dot_chart');
            const newIndex = (div.layout.annotations || []).length;
            if (newIndex > 0) {
                // Find if any annotation of the cluster appears before.
                // If so, remove the annotation.
                div.layout.annotations.forEach((ann, index) => {
                    if (ann.text === new_annotation.text) {
                        Plotly.relayout('cluster_dot_chart', 'annotations[' + index + ']', 'remove');
                        return;
                    }
                });
            }
            Plotly.relayout('cluster_dot_chart', 'annotations[' + newIndex + ']', new_annotation);
        });

        // Add chart hover over event to display cluster infor
        cluster_dot_chart.on('plotly_hover', function (data) {
            if (data.points.length > 0) {
                $('#hover_info').empty();
                const n = 10;
                const cluster_no = parseInt(data.points[0].data.name.split("#")[1]);
                const topics = get_cluster_topics(cluster_no, n);      // Get top 10 cluster topics
                const topic_text = topics.map(t => t['topic'] + ' (' + t['doc_ids'].length + ')' ).join(" ");
                // Add the cluster heading
                $('#hover_info').append($('<div class="h5">Cluster #' + cluster_no+' Top ' + n + ' topics</div>'));
                $('#hover_info').append($('<div>' + topic_text + '</div>'));
                $('#hover_info').focus();
            }
        });
        // Add unhover event to clear the text
        cluster_dot_chart.on('plotly_unhover', function (data) {
            $('#hover_info').empty();
        });
    }


    // Create the network graph using D3 library
    function _createUI() {
        $('#cluster_dot_chart').empty();
        $('#cluster_dot_chart').css('width', width).css('height', height);
        drawChart();
    }

    _createUI();
}

