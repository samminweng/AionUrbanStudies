// Create scatter graph
function ScatterGraph(cluster_approach, cluster_chart_data, cluster_topic_words) {
    const width = 600;
    const height = 650;
    // Find the maximal cluster number as total number of clusters
    const total_clusters = cluster_chart_data.map(c => c[cluster_approach+ '_Cluster']).reduce((p_value, c_value)=>{
        return (p_value >= c_value) ? p_value: c_value;
    }, 0);
    const clusters = cluster_topic_words[cluster_approach];

    // Optimal color pallets for 27 clusters
    // ref: https://mokole.com/palette.html
    const color_plates =[
        "#556b2f", "#8b4513", "#228b22", "#483d8b", "#008b8b", "#4682b4", "#000080", "#9acd32", "#daa520", "#8b008b",
        "#ff0000", "#ffff00", "#00ff00", "#8a2be2", "#00ff7f", "#dc143c", "#00ffff", "#0000ff", "#ff7f50", "#ff00ff",
        "#1e90ff", "#db7093", "#b0e0e6", "#ff1493", "#ee82ee", "#ffdab9"
    ];

    // Get the color of collocation
    const colors = function (cluster_no) {
        return color_plates[cluster_no];
    }

    // Draw google chart
    function drawChart() {
        let data = [];
        // Convert the clustered data into the format for Plotly js chart
        for (let cluster_no = 0; cluster_no <= total_clusters; cluster_no++) {
            const cluster_data = cluster_chart_data.filter(d => d[cluster_approach+'_Cluster'] === cluster_no);
            // const topic_words = Utility.get_topic_words_cluster(total_clusters, cluster_no);
            let data_point = {'x': [], 'y': [], 'label': []};
            for (const dot of cluster_data) {
                data_point['x'].push(dot.x);
                data_point['y'].push(dot.y);
                data_point['label'].push('Cluster: ' + cluster_no + ' Doc id: ' + dot.DocId);   // Tooltip label
            }

            let trace = {
                'x': data_point['x'], 'y': data_point['y'], 'text': data_point['label'],
                'name': 'Cluster ' + cluster_no, 'mode': 'markers', 'type': 'scatter',
                'marker': {color: colors(cluster_no)}
            };
            data.push(trace);
        }
        // Define the layout
        let layout = {
            // colorway: color_plates,
            autosize: true,
            width: width,
            height: height,
            margin: {
                l: 50,
                r: 50,
                b: 10,
                t: 10,
                pad: 3
            },
            // Plot the legend outside the
            showlegend: true,
            legend: {
                "orientation": "h"
                // x: 1,
                // xanchor: 'right',
                // y: 1
            },
            xaxis: {range: [7.2, 13.2]},
            yaxis: {range: [-0.2, 5.2]},
            annotations: []
        }

        // Get the cluster number
        Plotly.newPlot('doc_cluster', data, layout);
        // Add onclick event to the chart to display the cluster number
        const doc_cluster_div = document.getElementById('doc_cluster');
        doc_cluster_div.on('plotly_click', function (data) {
            const point = data.points[0];
            // Get the doc id from text
            const cluster_no = parseInt(point.data.name.split(" ")[1]);
            // Get cluster documents
            const num_docs = clusters.find(c => c['Cluster'] === cluster_no)['NumDocs'];
            // console.log(doc_cluster_div.layout.annotations)
            // Add an annotation to the clustered dots.
            const new_annotation = {
                x: point.xaxis.d2l(point.x), y: point.yaxis.d2l(point.y),
                bordercolor: point.fullData.marker.color,
                text: '<b>Cluster ' + cluster_no + '</b> <br>' +
                    '<i>' + num_docs + ' articles</i><br>'
            };
            Plotly.relayout('doc_cluster', 'annotations[0]', new_annotation);
        });
    }


    // Create the network graph using D3 library
    function _createUI() {
        $('#doc_cluster').empty();
        $('#doc_cluster').css('width', width).css('height', height);
        drawChart();

    }

    _createUI();
}

// // Get the number of documents for a collocation node
// function get_node_size(node_name) {
//     let num_doc = Utility.get_number_of_documents(node_name, collocation_data);
//     let radius = Math.sqrt(num_doc);
//     // let radius = num_doc / max_doc_ids * max_radius;
//     return Math.round(radius);  // Round the radius to the integer
// }
//
// // Get the number of documents for a link (between two terms
// function get_link_size(link) {
//     let source = link.source;
//     let target = link.target;
//     let occ = occurrence_data['occurrences'][source.id][target.id];
//     return Math.max(1.5, Math.sqrt(occ.length));
// }
//
// // Get the link color
// function get_link_color(link) {
//     let source = link.source;
//     let target = link.target;
//     let source_color = colors(source.name);
//     let target_color = colors(target.name);
//     if (source_color !== target_color) {
//         // Scale the color
//         return d3.schemeCategory10[7];
//     }
//     return source_color;
// }
