// Plotly Bar chart
// Ref: https://plotly.com/javascript/reference/bar/
function BarChart(group_data, sub_group_data, cluster, cluster_docs) {
    const width = 600;
    const height = 700;
    const d3colors = d3.schemeCategory10;
    group_data.sort((a, b) => a['Group'] - b['Group']);
    console.log(group_data);
    const min_group_id = group_data.reduce((pre, cur) => pre['Group'] < cur['Group'] ? pre : cur)['Group'];
    let thread = 2;
    if (min_group_id === 0) {
        thread = 1;
    }
    const data = create_graph_data();
    console.log(data);


    // Graph data
    function create_graph_data() {
        let data = [];
        // Re-order the groups to match with the order of the chart.

        for (let i =0; i< group_data.length; i++) {
            const group = group_data[i];
            const group_id = group['Group'];
            const group_name = "Topic#" + (group_id + thread);
            // Get the sub-group
            const sub_groups = sub_group_data.filter(g => g['Group'] === group_id);
            // create a trace
            let trace = {
                x: [], y: [], text: [],
                orientation: 'h', type: 'bar',
                name: group_name,
                textposition: 'auto',
                insidetextanchor: "start",
                insidetextfont: {
                    size: 14
                },
                outsidetextfont:{
                    size: 14
                },
                hovertemplate: '%{x} papers',
                marker: {
                    color: d3colors[group_id + 1]
                }
            };
            if(i > 0){
                trace['xaixs'] = "x" + (i+1)
                trace['yaxis'] = "y" + (i+1)
            }


            if (sub_groups.length > 0) {
                for (const sub_group of sub_groups) {
                    // console.log(sub_group);
                    const sub_group_id = sub_group['SubGroup'];
                    const title_words = sub_group['TitleWords'];
                    const num_docs = sub_group['NumDocs'];
                    trace['y'].push(group_name + "|" + sub_group_id);
                    trace['x'].push(num_docs);
                    trace['text'].push(title_words.join(", "))
                }
            } else {
                // Add the group
                const title_words = group['TitleWords'];
                const num_docs = group['NumDocs'];
                trace['y'].push(group_name + "#" + group_id);
                trace['x'].push(num_docs);
                trace['text'].push(title_words.join(", "))
            }
            data.push(trace);
        }

        return data;
    }


    function create_UI() {
        let layout = {
            xaxis: {
                title: 'Number of Papers',
                domain: [0, 1.0],
            },
            yaxis: {
                tickfont: {
                    size: 1,
                },
                domain: [0, 0.65]
            },
            yaxis2: {
                tickfont: {
                    size: 1,
                },
                domain: [0.7, 0.75]
            },
            yaxis3: {
                tickfont: {
                    size: 1,
                },
                domain: [0.8, 1.0]
            },
            width: width,
            height: height,
            autosize: false,
            showlegend: true,
            margin: {"l": 10, "r": 10, "t": 0},
            legend: {
                traceorder: 'reversed',
            },
            hovermode: 'closest',
            config: { responsive: true },
            grid: {
                rows: (group_data.length >=3 ? group_data.length: 3),   // Display three rows by default
                columns: 1,
                pattern: 'independent',
                roworder: 'bottom to top'}
        };
        console.log(data);
        // Plot bar chart
        Plotly.newPlot('key_phrase_chart', data, layout);

        const chart_element = document.getElementById('key_phrase_chart');

        // // Define the hover event
        chart_element.on('plotly_click', function (data) {
            $('#sub_group').empty();
            $('#doc_list').empty();
            const id = data.points[0].y;
            // Get the marker
            const marker = data.points[0].data.marker;
            const color = marker.color;
            // console.log(trace_index);
            // // Set the line width
            // marker.line.width = 3;
            // marker.line.color = 'black';
            // Plotly.restyle(chart_element, marker, [trace_index]);
            console.log(id);
            if (id.includes("#")) {
                const group_id = parseInt(id.split("#")[1]) - thread;
                // Get the sub-group
                if (id.includes("|")) {
                    const subgroup_id = parseInt(id.split("|")[1]);
                    const sub_group = sub_group_data.find(g => g['Group'] === group_id && g['SubGroup'] === subgroup_id);
                    if (sub_group) {
                        const view = new KeyPhraseView(sub_group, cluster_docs, color);
                    }
                } else {
                    const found = id.match(/#/g);
                    if (found && found.length === 2) {
                        // This indicates the groups has only one subgroup. so we use the group data.
                        // Get the group
                        const group = group_data.find(g => g['Group'] === group_id);
                        // Display the group
                        const view = new KeyPhraseView(group, cluster_docs, color);
                    }
                }
            }
        });// End of chart onclick event


    }

    create_UI();
}
