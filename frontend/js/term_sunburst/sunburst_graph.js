// Create Plotly sunburst graph to display Grouped key phrases and LDA Topics
// Ref: https://plotly.com/javascript/sunburst-charts/
// Ref: https://plotly.com/javascript/reference/sunburst/
function SunburstGraph(group_data, sub_group_data, cluster_no, cluster_docs) {
    const chart_div = $('#key_phrase_chart');
    // Convert the word_docs map to nodes/links
    const {labels, values, parents, ids, texts} = create_graph_data();
    console.log(ids);

    // Convert the json to the format of plotly graph
    function create_graph_data() {
        // Populate the groups of key phrases
        const root = "Cluster#"+cluster_no;
        let ids = [];
        let labels = [];
        let values = [];
        let parents = [];
        let texts = [];
        let total = 0;
        for(const group of group_data){
            const group_id = group['Group'];
            // Get the sub-group
            const sub_groups = sub_group_data.filter(g => g['Group'] === group_id);
            const group_name = "Group" + "#" + (group_id + 1);
            let group_total = group['NumDocs'];
            if(sub_groups.length > 0){
                group_total = sub_groups.reduce((pre, cur) => pre + cur['NumDocs'], 0);
            }
            total += group_total;
            ids.push(group_name);
            labels.push(group_name);
            values.push(group_total);
            texts.push(" (" + group['NumDocs'] + " papers)")
            parents.push(root);
            if(sub_groups.length > 0){
                for(const sub_group of sub_groups){
                    const sub_group_id =  group_name + "-" + sub_group['SubGroup'];
                    const sub_group_label = sub_group['TitleWords'].join(",<br>");
                    const sub_group_total = sub_group['NumDocs'];
                    ids.push(sub_group_id);
                    labels.push(sub_group_label);
                    values.push(sub_group_total);
                    texts.push(" (" + sub_group_total + " papers)");
                    parents.push(group_name);
                }
            }else{
                // Add the group
                const sub_group_id = group_name + "#" +group_id;
                const sub_group_label = group['TitleWords'].join(",<br>");
                const sub_group_total = group['NumDocs'];
                ids.push(sub_group_id);
                labels.push(sub_group_label);
                values.push(sub_group_total);
                texts.push(" (" + sub_group_total + " papers)");
                parents.push(group_name);
            }
        }
        // Add the root
        ids.push(root);
        labels.push(root);
        values.push(total);
        parents.push("");
        texts.push(" (" + cluster_docs.length + " papers)");

        return {labels: labels, values: values, parents: parents, ids:ids, texts: texts};
    }

    // Create the sunburst
    function create_sunburst_graph() {
        chart_div.empty();
        const data = [{
            "type": "sunburst",
            "maxdepth": 3,
            "ids": ids,
            "labels": labels,
            "parents": parents,
            "text": texts,
            "values": values,
            "leaf": {"opacity": 0.7},
            "marker": {"line": {"width": 2}},
            "branchvalues": 'total',
            "outsidetextfont": {size: 24, color: "#377eb8"},
            "insidetextfont": {size: 16},
            'insidetextorientation': "horizontal",
            // Label text orientation
            "textposition": 'inside',
            // "hovertemplate": "<b>%{label}</b> %{text}",
            'hoverinfo': 'label+text',
        }];

        const layout = {
            "margin": {"l": 10, "r": 10, "b": 10, "t": 10},
            hovermode: 'closest',
            config: { responsive: true }
            // "sunburstcolorway": D3Colors
        };
        const chart_id = chart_div.attr('id');
        Plotly.newPlot(chart_id, data, layout, {showSendToCloud: true})
        const chart_element = document.getElementById(chart_id);

        // // Define the hover event
        chart_element.on('plotly_click', function(data){
            const id = data.points[0].id;
            if(id.includes("#")){
                if(id.includes("-")){
                    const group_id = parseInt(id.split("#")[1]) - 1;
                    const subgroup_id = parseInt(id.split("-")[1])
                    // const group = group_data.find(g => g['Group'] === group_id);
                    const sub_group = sub_group_data.find(g => g['Group'] === group_id && g['SubGroup'] === subgroup_id);
                    if(sub_group){
                        const sub_group_table = new KeyPhraseTable(sub_group, cluster_docs);
                    }
                }else{
                    // const list_view = new KeyPhraseListView(group, sub_groups, cluster_docs);
                }
            }
        });// End of chart onclick event
    }
    // Create the sunburst graph using D3 library
    function createUI() {
        try {
            create_sunburst_graph();
        } catch (error) {
            console.error(error);
        }
    }

    createUI();
}
