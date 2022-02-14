// Plotly Bar chart
// Ref: https://plotly.com/javascript/reference/bar/
function BarChart(group_data, sub_group_data, cluster, cluster_docs) {
    const width = 600;
    const height = 700;
    const d3colors = d3.schemeCategory10;
    group_data.sort((a, b) => a['Group'] - b['Group']);
    console.log(group_data);    // Three main groups of key phrases
    console.log(sub_group_data);    // Each main group contain a number of sub_groups
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
            const group_name = "Group#" + (group_id + thread);
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
                hovertemplate: '%{text}',
                marker: {
                    color: d3colors[group_id + 1]
                }
            };
            if(i > 0){
                trace['xaixs'] = "x" + (i+1)
                trace['yaxis'] = "y" + (i+1)
            }else{
                trace['xaixs'] = "x"
                trace['yaxis'] = "y"
            }

            if (sub_groups.length > 0) {
                for (const sub_group of sub_groups) {
                    // console.log(sub_group);
                    const sub_group_id = sub_group['SubGroup'];
                    const title_words = sub_group['TitleWords'];
                    const num_docs = sub_group['NumDocs'];
                    trace['y'].push(group_name + "|" + sub_group_id);
                    trace['x'].push(num_docs);
                    trace['text'].push('<b>' + title_words.join(", ") + '</b>')
                }
            } else {
                // Add the group
                const title_words = group['TitleWords'];
                const num_docs = group['NumDocs'];
                trace['y'].push(group_name + "#" + group_id);
                trace['x'].push(num_docs);
                trace['text'].push('<b>' + title_words.join(", ") + '</b>')
            }
            data.push(trace);
        }

        return data;
    }

    function populate_y_axis_domain(layout){
        let total_sub_groups = 0;
        // Go through each main group
        for (let i =0; i< group_data.length; i++) {
            const group = group_data[i];
            const group_id = group['Group'];
            // Get the sub-group
            const sub_groups = sub_group_data.filter(g => g['Group'] === group_id);
            if(sub_groups.length > 0){
                total_sub_groups += sub_groups.length;
            }else{
                // The group does not have a sub-group
                total_sub_groups += 1;
            }
        }
        // Get the portion of each sub-group
        let portion = (1.0 / total_sub_groups * 0.85);
        let gap = 0.02;
        if(group_data.length > 1){
            gap = (1.0 - (portion * total_sub_groups))/(group_data.length-1);    // Gap between different groups
        }
        let cur_domain = 0.0;
        for(let i=0; i < group_data.length; i++){
            const group = group_data[i];
            const group_id = group['Group'];
            // Get the sub-group
            const sub_groups = sub_group_data.filter(g => g['Group'] === group_id);
            // Add one base portion
            let next_domain = cur_domain + portion;
            if(sub_groups.length > 0){
                // The domain Proportion to the number of sub-groups
                next_domain = cur_domain + (sub_groups.length * portion);
            }
            // Get axis name
            const axis_name =  (i > 0 ? "yaxis" + (i+1) : "yaxis");
            layout[axis_name] ={
                tickfont: {
                    size: 1,
                },
                domain: [cur_domain, next_domain]
            }
            if(i < group_data.length -1){
                cur_domain = next_domain + gap;// Add the gap to separate different groups
            }else{
                cur_domain = next_domain;
            }
        }
        return layout;
    }



    function create_UI() {
        let layout = {
            xaxis: {
                title: 'Number of Papers',
                domain: [0, 1.0],
            },
            width: width,
            height: height,
            autosize: true,
            showlegend: true,
            margin: {"l": 10, "r": 10, "t": 10},
            legend: { traceorder: 'reversed'},
            grid: {
                rows: (group_data.length >=3 ? group_data.length: 3),   // Display three rows by default
                columns: 1,
                pattern: 'independent',
                roworder: 'bottom to top'
            },

        };
        console.log(data);
        layout = populate_y_axis_domain(layout);
        console.log(layout);
        const config = {
            displayModeBar: false // Hide the floating bar
        }
        // Plot bar chart
        Plotly.newPlot('key_phrase_chart', data, layout, config);
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
