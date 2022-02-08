// Create Plotly sunburst graph to display Grouped key phrases and LDA Topics
// Ref: https://plotly.com/javascript/sunburst-charts/
// Ref: https://plotly.com/javascript/reference/sunburst/
function SunburstGraph(group_data, sub_group_data, cluster_no, cluster_docs) {
    const chart_div = $('#key_phrase_chart');
    // Convert the word_docs map to nodes/links
    const total = group_data.reduce((pre, cur) => pre + cur['NumPhrases'], 0);
    const {labels, values, parents, ids, texts} = create_graph_data();
    console.log(ids);

    // Convert the json to the format of plotly graph
    function create_graph_data() {
        // Populate the groups of key phrases
        const root = "Cluster#"+cluster_no;
        let ids = [root];
        let labels = [root];
        let values = [total];
        let parents = [""];
        for(const group of group_data){
            const group_total = group['NumPhrases'];
            const group_id = group['Group'];
            const group_name = "Group" + "#" + (group_id + 1);
            ids.push(group_name);
            labels.push(group_name);
            values.push(group_total);
            parents.push(root);
            // Get the sub-group
            const sub_groups = sub_group_data.filter(g => g['Group'] === group_id);
            if(sub_groups.length > 0){
                for(const sub_group of sub_groups){
                    const sub_group_id =  group_name + "#" + sub_group['SubGroup'];
                    const sub_group_label = sub_group['TitleWords'].join(",<br>");
                    const sub_group_total = sub_group['NumPhrases'];
                    ids.push(sub_group_id);
                    labels.push(sub_group_label);
                    values.push(sub_group_total);
                    parents.push(group_name);
                }
            }else{
                // Add the group
                const sub_group_id = group_name + "#" +group_id;
                const sub_group_label = group['TitleWords'].join(",<br>");
                const sub_group_total = group['NumPhrases'];
                ids.push(sub_group_id);
                labels.push(sub_group_label);
                values.push(sub_group_total);
                parents.push(group_name);
            }

        }
        return {labels: labels, values: values, parents: parents, ids:ids};
    }

    // Create the group header
    function create_group_header(group){
        // Added the header of key-phrase group
        const header = $('<div class="row"> </div>');
        const group_id = group['Group'] + 1;
        // // Add Cluster
        // header.append($('<div class="col"> Cluster: #' + cluster_no + ' </div>'));
        // Add group no
        header.append($('<div class="col"> Group: #' + group_id + ' </div>'));
        // Add the num of phrases
        header.append($('<div class="col"> Total Pharses: ' + group['Key-phrases'].length + ' </div>'));
        // Add the num of papers
        header.append($('<div class="col"> Total Papers: ' + group['DocIds'].length + ' </div>'));

        // Added to UI
        $('#key_phrase_header').empty();
        $('#key_phrase_header').append(header);
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
            const group_id = parseInt(id.split("#")[1]) - 1;
            const group = group_data.find(g => g['Group'] === group_id);
            if(group){
                const sub_groups = sub_group_data.filter(g => g['Group'] === group_id);
                if(sub_groups.length > 0){
                    // Create a list view to display all the sub-groups
                    const list_view = new KeyPhraseListView(group, sub_groups, total, cluster_docs);
                }else{
                    // Create a list view to display all the sub-groups
                    const list_view = new KeyPhraseListView(group, [group], total, cluster_docs);
                }
                create_group_header(group);
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
