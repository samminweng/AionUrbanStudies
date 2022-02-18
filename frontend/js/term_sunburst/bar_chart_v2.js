// Plotly Bar chart
// Ref: https://plotly.com/javascript/reference/bar/
function BarChart(group_data, sub_group_data, cluster, cluster_docs) {
    const width = 500;
    const d3colors = d3.schemeCategory10;
    group_data.sort((a, b) => a['Group'] - b['Group']);
    // console.log(group_data);    // Three main groups of key phrases
    // console.log(sub_group_data);    // Each main group contain a number of sub_groups
    const min_group_id = group_data.reduce((pre, cur) => pre['Group'] < cur['Group'] ? pre : cur)['Group'];
    let thread = 2;
    if (min_group_id === 0) {
        thread = 1;
    }

    // Graph data for a group
    function create_graph_data(group, max_size) {
        let data = [];
        // Re-order the groups to match with the order of the chart.
        // const group = group_data[i];
        const group_id = group['Group'];
        const group_name = "Group#" + (group_id + thread);
        // Get the sub-group
        const sub_groups = sub_group_data.filter(g => g['Group'] === group_id);
        // create a trace
        let trace = {
            x: [], y: [], text: [],
            orientation: 'h', type: 'bar',
            name: group_name,
            textposition: 'insides',
            insidetextanchor: "start",
            insidetextfont: {
                size: 14
            },
            outsidetextfont: {
                size: 14
            },
            hovertemplate: '%{x} papers',
            marker: {
                color: d3colors[group_id + 1],
                line: {
                    color: 'black',
                    width: 1
                }
            },
            opacity: 0.5,
        };
        let comp_trace ={
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

        if (sub_groups.length > 0) {
            for (const sub_group of sub_groups) {
                // console.log(sub_group);
                const sub_group_id = sub_group['SubGroup'];
                const key_phrases = sub_group['Key-phrases'];
                // Get the title words of a sub-group
                const title_words = Utility.collect_title_words(key_phrases);
                sub_group['TitleWords'] = title_words;
                // Update the title word of a sub-group;
                const num_docs = sub_group['NumDocs'];
                trace['y'].push(group_name + "|" + sub_group_id);
                trace['x'].push(num_docs);
                // trace['text'].push('<b>' + title_words.slice(0, 3).join(", ") + '</b>');
                comp_trace['y'].push(group_name + "|" + sub_group_id);
                comp_trace['x'].push(max_size - num_docs);
                // comp_trace['text'].push();
                annotations.push({
                    x: 0.0,
                    y: group_name + "|" + sub_group_id,
                    text: '<b>' + title_words.slice(0, 3).join(", ") + '</b>',
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

            }
        } else {
            // Add the group
            // const title_words = group['TitleWords'];
            const title_words = Utility.collect_title_words(group['Key-phrases']);
            group['TitleWords'] = title_words;
            const num_docs = group['NumDocs'];
            trace['y'].push(group_name + "#" + group_id);
            trace['x'].push(num_docs);
            // trace['text'].push('<b>' + title_words.slice(0, 3).join(", ") + '</b>');
            comp_trace['y'].push(group_name + "#" + group_id);
            comp_trace['x'].push(max_size - num_docs);
            annotations.push({
                x: 0.0,
                y: group_name + "#" + group_id,
                text: '<b>' + title_words.join(", ").substring(0, 50) + '...</b>',
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
        }
        data.push(trace);
        data.push(comp_trace);

        return [data, annotations];
    }

    // Create a bar chart for each group
    function create_bar_chart(group, chart_id, max_size){
        const x_domain = [0, max_size];
        const [graph_data, annotations] = create_graph_data(group, max_size);
        const group_id = group['Group'];
        const group_name = "Group#" + (group_id + thread);
        // Get the sub-group
        const sub_groups = sub_group_data.filter(g => g['Group'] === group_id);
        const height = 50;
        const gap = 80;
        let graph_height = (sub_groups.length >0 ? height * sub_groups.length + gap: height*2);

        // Graph layout
        let layout = {
            width: width,
            height: graph_height,
            xaxis: {range: x_domain},
            showlegend: false,
            margin: {"l": 10, "r": 10, "t": 0, "b": height},
            legend: { traceorder: 'reversed'},
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
            $('#sub_group').empty();
            $('#doc_list').empty();
            const id = data.points[0].y;
            // Get the marker
            const marker = data.points[0].data.marker;
            const color = marker.color;
            // console.log(id);
            if (id.includes("#")) {
                const group_id = parseInt(id.split("#")[1]) - thread;
                // Get the sub-group
                if (id.includes("|")) {
                    const subgroup_id = parseInt(id.split("|")[1]);
                    const sub_group = sub_group_data.find(g => g['Group'] === group_id && g['SubGroup'] === subgroup_id);
                    if (sub_group) {
                        const word_chart = new WordBubbleChart(sub_group, cluster_docs, color);
                        const view = new KeyPhraseView(sub_group, cluster_docs, color);
                    }
                } else {
                    const found = id.match(/#/g);
                    if (found && found.length === 2) {
                        // This indicates the groups has only one subgroup. so we use the group data.
                        // Get the group
                        const group = group_data.find(g => g['Group'] === group_id);
                        // Display the group
                        const word_chart = new WordBubbleChart(group, cluster_docs, color);
                        const view = new KeyPhraseView(group, cluster_docs, color);
                    }
                }
            }
        });// End of chart onclick event
    }

    // Main entry
    function create_UI() {
        $('#key_phrase_chart').empty();
        let max_size = 0;
        for(let i =0; i < group_data.length; i++){
            const chart_id = 'chart_' + i;
            // Create a div
            const graph_div = $('<div id="' + chart_id +'" class="col"></div>')
            $('#key_phrase_chart').append($('<div class="row"></div>').append(graph_div));
            const group = group_data[i];
            const group_id = group['Group'];
            const sub_groups = sub_group_data.filter(g => g['Group'] === group_id);
            if(sub_groups.length >0 ){
                for(const sub_group of sub_groups){
                    const num_docs = sub_group['NumDocs'];
                    max_size = Math.max(num_docs, max_size);
                }
            }else{
                max_size = Math.max(max_size, group['NumDocs']);
            }
        }
        // max_size = max_size + 1;
        // console.log("max_size", max_size);
        // Display the bar chart for each group
        for(let i =0; i < group_data.length; i++){
            const chart_id = 'chart_' + i;
            // Get the group
            const group = group_data[i];
            create_bar_chart(group, chart_id, max_size);
        }
        // // For development only
        // Create a term chart of sub_group
        const group_id = group_data[0]['Group'];
        const sub_groups = sub_group_data.filter(g => g['Group'] === group_id);
        if(sub_groups.length > 0){
            const view = new KeyPhraseView(sub_groups[0], cluster_docs, d3colors[0]);
            const word_chart = new WordBubbleChart(sub_groups[0], cluster_docs, d3colors[0]);
        }else{
            const view = new KeyPhraseView(group_data[0], cluster_docs, d3colors[0]);
            const word_chart = new WordBubbleChart(group_data[0], cluster_docs, d3colors[0]);
        }


    }

    create_UI();
}
