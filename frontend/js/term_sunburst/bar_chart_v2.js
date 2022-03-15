// Plotly Bar chart
// Ref: https://plotly.com/javascript/reference/bar/
function BarChart(group_data, cluster, cluster_docs) {
    const width = 500;
    const d3colors = d3.schemeCategory10;
    group_data.sort((a, b) => a['Group'] - b['Group']);

    // Graph data for a group
    function create_graph_data(group, group_id, max_size) {
        let data = [];
        // Re-order the groups to match with the order of the chart.
        const group_name = "Group#" + group_id;
        // create a trace
        let trace = {
            x: [], y: [], text: [],
            orientation: 'h', type: 'bar',
            name: group_name,
            textposition: 'none',
            hoverinfo: "text",
            marker: {
                color: d3colors[group_id],
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
        const MAXLENGTH = 50;

        // Add the group
        console.log("group", group);
        const topic_words = group['topic_words'];
        const key_phrases = group['key-phrases'];

        const num_docs = group['NumDocs'];
        trace['y'].push(group_name);
        trace['x'].push(num_docs);
        trace['text'].push('<b>' + key_phrases.length + ' key phrases, ' + num_docs + ' papers</b>');
        // trace['text'].push('<b>' + title_words.slice(0, 3).join(", ") + '</b>');
        comp_trace['y'].push(group_name);
        comp_trace['x'].push(max_size - num_docs);
        annotations.push({
            x: 0.0,
            y: group_name,
            text: '<b>' + topic_words.join(", ").substring(0, MAXLENGTH) + '...</b>',
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
    }

    // Create a bar chart for each group
    function create_bar_chart(group, group_id, max_size) {
        const x_domain = [0, max_size];
        const [graph_data, annotations] = create_graph_data(group, group_id, max_size);
        const height = 50;
        let graph_height = height * 2;
        // Graph layout
        let layout = {
            width: width,
            height: graph_height,
            xaxis: {range: x_domain},
            showlegend: false,
            margin: {"l": 10, "r": 10, "t": 0, "b": height},
            legend: {traceorder: 'reversed'},
            barmode: 'stack',
            annotations: annotations
        };
        // console.log(graph_data);
        // console.log(layout);
        const config = {
            displayModeBar: false // Hide the floating bar
        }
        const chart_id = 'chart_' + group_id;
        // Plot bar chart
        Plotly.newPlot(chart_id, graph_data, layout, config);
        const chart_element = document.getElementById(chart_id);
        // // Define the hover event
        chart_element.on('plotly_click', function (data) {
            $('#term_occ_chart').empty();
            $('#group').empty();
            $('#doc_list').empty();
            const id = data.points[0].y;
            // Get the marker
            const marker = data.points[0].data.marker;
            const color = marker.color;
            // console.log(id);
            if (id.includes("#")) {
                const group_id = parseInt(id.split("#")[1]);
                const found = id.match(/#/g);
                if (found) {
                    // This indicates the groups has only one subgroup. so we use the group data.
                    // Get the group
                    const group = group_data[group_id];
                    // Display the group
                    const word_chart = new WordBubbleChart(group, cluster_docs, color);
                    const view = new KeyPhraseView(group, cluster_docs, color);
                }
            }
        });// End of chart onclick event
    }

    // Main entry
    function create_UI() {
        $('#key_phrase_chart').empty();
        let max_size = 0;
        for (let i = 0; i < group_data.length; i++) {
            const group_id = i;
            const chart_id = 'chart_' + group_id;
            // Create a div
            const graph_div = $('<div id="' + chart_id + '" class="col"></div>')
            $('#key_phrase_chart').append($('<div class="row"></div>').append(graph_div));
            const group = group_data[i];
            max_size = Math.max(max_size, group['NumDocs']);
        }
        // max_size = max_size + 1;
        // console.log("max_size", max_size);
        // Display the bar chart for each group
        for (let i = 0; i < group_data.length; i++) {
            const group_id = i;
            // Get the group
            const group = group_data[group_id];
            create_bar_chart(group, group_id, max_size);
        }
        // // For development only
        // Create a term chart of sub_group
        const group = group_data[0];
        const group_id = group['Group'];
        const view = new KeyPhraseView(group, cluster_docs, d3colors[0]);
        const word_chart = new WordBubbleChart(group, cluster_docs, d3colors[0]);
    }

    create_UI();
}
