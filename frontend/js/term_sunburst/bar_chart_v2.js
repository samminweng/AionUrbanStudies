// Plotly Bar chart
// Ref: https://plotly.com/javascript/reference/bar/
function BarChart(keyword_clusters, cluster, cluster_docs) {
    const width = 500;
    const colors = ["#1f77b4","#ff7f0e","#2ca02c","#d62728","#9467bd",
                    "#8c564b","#e377c2","#7f7f7f","#bcbd22","#17becf"];

    // Create a bar chart for each group
    function create_bar_chart(keyword_cluster, group_id, max_size, chart_id) {
        // Graph data for a group
        const create_graph_data = function(keyword_cluster, group_id, max_size) {
            let data = [];
            // Re-order the groups to match with the order of the chart.
            const group_name = "Topic " + group_id;
            // create a trace
            let trace = {
                x: [], y: [], text: [],
                orientation: 'h', type: 'bar',
                name: group_name,
                yaxis: group_name,
                textposition: 'none',
                hoverinfo: "text",
                marker: {
                    color: colors[group_id],
                    line: {
                        color: 'black',
                        width: 1
                    }
                },
                opacity: 0.5,
            };
            let comp_trace = {
                x: [], y: [], text: [],
                orientation: 'h', type: 'bar',
                name: group_name,
                marker: {
                    color: 'white',
                    line: {
                        color: 'black',
                        width: 1
                    }
                },
                opacity: 0.5,
                hoverinfo: 'none',
            }
            // Ref: https://plotly.com/javascript/reference/layout/annotations/
            // A text can
            let annotations = [];
            // Add the topic words
            const topic_words = keyword_cluster['TopicWords'];
            const key_phrases = keyword_cluster['Key-phrases'];
            const num_docs = keyword_cluster['NumDocs'];
            trace['y'].push(group_name);
            trace['x'].push(num_docs);
            trace['text'].push('<b>' + num_docs + ' articles</b> ('+ key_phrases.length + ' keywords)');
            comp_trace['y'].push(group_name);
            comp_trace['x'].push(max_size - num_docs);
            annotations.push({
                x: 0.0,
                y: group_name,
                text: '' + topic_words.join(", ") + '...',
                font: {
                    family: 'Arial',
                    size: 14,
                    color: 'black'
                },
                xref: 'paper',
                xanchor: 'left',
                align: 'left',
                showarrow: false
            })

            data.push(trace);
            data.push(comp_trace);

            return [data, annotations];
        };

        const x_domain = [0, max_size];
        const [graph_data, annotations] = create_graph_data(keyword_cluster, group_id, max_size);
        const height = 45;
        let graph_height = height * 2;
        // Graph layout
        let layout = {
            width: width,
            height: graph_height,
            xaxis: {range: x_domain},
            showlegend: false,
            margin: {"l": 40, "r": 10, "t": 0, "b": height},
            legend: {traceorder: 'reversed'},
            barmode: 'stack',
            annotations: annotations
        };
        // console.log(graph_data);
        // console.log(layout);
        const config = {
            displayModeBar: false // Hide the floating bar
        }
        // Plot bar chart
        Plotly.newPlot(chart_id, graph_data, layout, config);
        const chart_element = document.getElementById(chart_id);
        // // Define the hover event
        chart_element.on('plotly_click', function (data) {
            $('#term_occ_chart').empty();
            $('#group').empty();
            $('#doc_list').empty();
            const id = data.points[0].y;
            // console.log(id);
            if (id.includes("Topic")) {
                const group_id = parseInt(id.split("Topic")[1]);
                // Get the marker color
                const color = data.points[0].data.marker.color;
                // This indicates the groups has only one subgroup. so we use the group data.
                // Get the group
                const group = keyword_clusters.find(c => c['Group'] === group_id);
                // Display the group
                const word_chart = new WordBubbleChart(group, cluster_docs, color);
                const view = new KeyPhraseView(group, cluster_docs, 0);
            }
        });// End of chart onclick event
    }

    // Main entry
    function create_UI() {
        $('#key_phrase_chart').empty();
        const max_size = cluster_docs.length;
        for (const keyword_cluster of keyword_clusters) {
            const group_id = keyword_cluster['Group'];
            const chart_id = 'chart_' + group_id;
            // Create a div
            const graph_div = $('<div id="' + chart_id + '" class="col"></div>')
            $('#key_phrase_chart').append($('<div class="row"></div>').append(graph_div));
            create_bar_chart(keyword_cluster, group_id, max_size, chart_id);
        }

        // // For development only
        // Create a term chart of group
        const keyword_cluster = keyword_clusters[0];
        const view = new KeyPhraseView(keyword_cluster, cluster_docs, 0);
        const word_chart = new WordBubbleChart(keyword_cluster, cluster_docs, colors[1]);
    }

    create_UI();
}
