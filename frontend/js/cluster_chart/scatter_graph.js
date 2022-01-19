// Create scatter graph
function ScatterGraph(is_hide, corpus_data, cluster_topic_key_phrases) {
    const width = 600;
    const height = 700;
    // Optimal color pallets for 30 colors from http://vrl.cs.brown.edu/color
    // citation:
    // @article{gramazio-2017-ccd,
    //   author={Gramazio, Connor C. and Laidlaw, David H. and Schloss, Karen B.},
    //   journal={IEEE Transactions on Visualization and Computer Graphics},
    //   title={Colorgorical: creating discriminable and preferable color palettes for information visualization},
    //   year={2017}
    // }
    const color_plates = ["#32964d", "#85e5dd", "#0e1e22", "#6297b3", "#18519b", "#e3a3e7", "#3c0223", "#be3acd",
                          "#3a187b", "#dee0ff", "#8f52a5", "#76dd78", "#0b5313", "#bbcf7a", "#7c8a4f", "#a2fa12",
                          "#744822", "#e7ad79", "#d01f18", "#e41a72", "#faf81c", "#ff8889", "#fbbd13", "#270fe2",
                          "#315bf3"];
    // Find the maximal cluster number as total number of clusters
    const total_clusters = corpus_data.map(c => c['Cluster']).reduce((p_value, c_value) => {
        return (p_value >= c_value) ? p_value : c_value;
    }, 0) + 1;
    console.log(total_clusters);
    // Get top N topics of a cluster
    function get_cluster_topics(cluster_no, n) {
        // Cluster top 5 topics
        const topics = cluster_topic_key_phrases.find(c => c['Cluster'] === cluster_no)['Topics'].slice(0, n);
        return topics;
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
        const initial_cluster = (is_hide) ? 0 : -1;
        // Convert the clustered data into the format for Plotly js chart
        for (let cluster_no = initial_cluster; cluster_no < total_clusters; cluster_no++) {
            const cluster_docs = corpus_data.filter(d => d['Cluster'] === cluster_no);
            if (cluster_docs.length > 0) {
                let data_point = {'x': [], 'y': [], 'label': []};
                for (const doc of cluster_docs) {
                    data_point['x'].push(doc.x);
                    data_point['y'].push(doc.y);
                    const topics = get_cluster_topics(cluster_no, 5);
                    const topic_text = topics.map(t => t['topic']).join("<br>");
                    if(cluster_no !== -1){
                        // Tooltip label displays top 5 topics
                        data_point['label'].push('<b>Cluster #' + cluster_no + '</b> ('+ cluster_docs.length  + ' papers)<br>' + topic_text);
                    }else{
                        // Tooltip label displays top 5 topics
                        data_point['label'].push('<b>Outlier</b> ('+ cluster_docs.length  + ' papers)<br>' + topic_text);
                    }
                }
                const trace_name = (cluster_no !== -1)? 'Cluster #' + cluster_no: "Outliers";
                // Trace setting
                let trace = {
                    'x': data_point['x'], 'y': data_point['y'], 'text': data_point['label'],
                    'name': trace_name, 'mode': 'markers', 'type': 'scatter',
                    'marker': {color: colors(cluster_no)}, opacity: opacity(cluster_no),
                    'hovertemplate': '%{text}'
                };
                data_points.push(trace);
            }
        }
        return data_points;
    }

    // Display top topics above the chart
    function display_top_topics(cluster_no){
        $('#hover_info').empty();
        const n = 10;
        const topics = get_cluster_topics(cluster_no, n);      // Get top 10 cluster topics
        const topic_text = topics.map(t => t['topic']).join(", ");
        // Add the cluster heading
        if(cluster_no !== -1){
            $('#hover_info').append($('<div class="h5">Cluster #' + cluster_no+' Top ' + n + ' topics</div>'));
        }else{
            $('#hover_info').append($('<div class="h5">Outlier Top ' + n + ' topics</div>'));
        }
        $('#hover_info').append($('<div>' + topic_text + '</div>'));
        $('#hover_info').focus();
    }

    // Draw google chart
    function drawChart() {
        const data_points = convert_cluster_data_to_data_points();
        // Define the layout
        const option = {
            autosize: true,
            width: width,
            height: height,
            // Set the graph margin
            margin: {
                l: 30,
                r: 30,
                b: 30,
                t: 30
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
            config: { responsive: true }
        }

        // Get the cluster number
        Plotly.newPlot('cluster_chart', data_points, option);
        const cluster_chart = document.getElementById('cluster_chart');
        // // Add chart onclick to toggle annotation
        cluster_chart.on('plotly_click', function (data) {
            const point = data.points[0];
            // Get the doc id from text
            let cluster_no = -1;
            if(point.data.name.includes('#')){
                cluster_no = parseInt(point.data.name.split("#")[1]);
            }
            // display_top_10_topics(cluster_no);
            // Create a list of cluster doc
            // const cluster = clusters.find(c => c['Cluster'] === cluster_no);
            const cluster_text = (cluster_no !== -1) ? 'Cluster #' + cluster_no : "Outliers";
            const cluster_doc_list = new ClusterDocList(cluster_no, corpus_data, cluster_topic_key_phrases);
            // Add an annotation to the clustered dots
            const new_annotation = {
                x: point.xaxis.d2l(point.x),
                y: point.yaxis.d2l(point.y),
                bordercolor: point.fullData.marker.color,
                text: '<b>' + cluster_text + '</b>'
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
        cluster_chart.on('plotly_hover', function(data){
            if(data.points.length> 0){
                const point = data.points[0];
                let cluster_no = -1;
                if(point.data.name.includes('#')){
                    cluster_no = parseInt(point.data.name.split("#")[1]);
                }
                display_top_topics(cluster_no);
            }
        }).on('plotly_unhover', function(data){
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

