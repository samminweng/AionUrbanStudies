// Create scatter graph
function ScatterGraph(total_clusters, doc_cluster_data) {
    const width = 600;
    const height = 600;
    // Get the color of collocation
    const colors = function (cluster) {
        // let group = Utility.get_group_number(collocation)
        return d3.schemeCategory10[cluster];
    }

    // Draw google chart
    function drawChart() {
        let data = [];
        for (let cluster_no = 0; cluster_no < total_clusters; cluster_no++) {
            const cluster_data = doc_cluster_data.filter(d => d['Cluster'] === cluster_no);
            // const topic_words = Utility.get_topic_words_cluster(total_clusters, cluster_no);
            let data_point = {'x': [], 'y': [], 'label': []};
            for (const dot of cluster_data) {
                data_point['x'].push(dot.x);
                data_point['y'].push(dot.y);
                data_point['label'].push("Doc id: " + dot.DocId);   // Tooltip label
            }

            let trace = {
                'x': data_point['x'], 'y': data_point['y'], 'text': data_point['label'],
                'name': 'Cluster ' + cluster_no, 'mode': 'markers', 'type': 'scatter'
            };
            data.push(trace);
        }
        // Define the layout
        let layout = {
            xaxis: { range: [0, 10]},
            yaxis: { range: [0, 10]},
            annotations: []
        }

        // Get the cluster number
        Plotly.newPlot('doc_cluster', data, layout);
        // Add event
        const doc_cluster_div = document.getElementById('doc_cluster');
        doc_cluster_div.on('plotly_click', function(data){
            const point = data.points[0];
            // Get the doc id from text
            const cluster_no = parseInt(point.data.name.split(" ")[1]);
            // Get cluster documents
            const {topic_words, documents} = Utility.get_cluster_documents(total_clusters, cluster_no);

            const doc_cluster_div = document.getElementById('doc_cluster');
            console.log(doc_cluster_div.layout.annotations)
            // Add an annotation to the clustered dots.
            // if(doc_cluster_div.layout.annotations.length < total_clusters){
                const new_annotation = {x: point.xaxis.d2l(point.x), y: point.yaxis.d2l(point.y),
                    bordercolor: point.fullData.marker.color,
                    borderwidth: 3,
                    borderpad: 4,
                    text: '<b>Cluster ' + cluster_no + '</b><br>' +
                        '<i>' + topic_words[0] + '</i><br>' +
                        '<i>' + topic_words[1] + '</i>'};
                Plotly.relayout('doc_cluster', 'annotations['+cluster_no+']', new_annotation);
            // }
            // console.log(topic_words);
            // Display the clustered documents
            // const doc_list_view = new DocumentListView(cluster_no, topic_words, documents);
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
