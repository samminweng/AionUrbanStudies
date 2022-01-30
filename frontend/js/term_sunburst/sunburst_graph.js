// Create Plotly sunburst graph to display Grouped key phrases and LDA Topics
// Ref: https://plotly.com/javascript/sunburst-charts/
// Ref: https://plotly.com/javascript/reference/sunburst/
function SunburstGraph(group_data, cluster_no, is_key_phrase) {
    const chart_div = (is_key_phrase? $('#key_phrase_chart'): $('#lda_topic_chart'));
    // Convert the word_docs map to nodes/links
    const {total, group_list} = compute_total(group_data);
    const {labels, values, parents, ids} = create_graph_data();
    console.log(ids);
    const width = 600;
    const height = 600;
    const D3Colors = d3.schemeCategory10;

    // Count the count of a groups
    function compute_total(key_phrase_groups){
        let total = 0;
        let group_list = [];
        for(const group of key_phrase_groups){
            let group_total = 0;
            for(const [top_word, doc_ids] of Object.entries(group['word_docIds'])){
                group_total += doc_ids.length;
                total += doc_ids.length;
            }
            group_list.push({group: group, total: group_total});
        }
        // Sort by the total
        group_list.sort((a, b) => b['total'] - a['total'])
        return {total: total, group_list: group_list};
    }

    // Convert the json to the format of plotly graph
    function create_graph_data() {
        // Populate the groups of key phrases
        const root = "Cluster#"+cluster_no;
        let ids = [root];
        let labels = [root];
        let values = [total];
        let parents = [""];
        for(let i=0; i < group_list.length; i++){
            const group = group_list[i]['group'];
            const group_total = group_list[i]['total'];
            const prefix = (is_key_phrase ? "Group": "Topic");
            const group_name = prefix + " #" + (i + 1);
            const group_score = group['score'].toFixed(2);
            ids.push(group_name);
            labels.push(group_name + " <br>score:" + group_score);
            values.push(group_total);
            parents.push(root);
            const top_words = group['top_words'];
            for(const top_word of top_words){
                const num_docs = group['word_docIds'][top_word].length;
                const w_id =  group_name + " - " + top_word;
                let w_label = top_word;
                const w_array = top_word.split(" ");
                if(w_array.length > 2){
                    w_label = w_array[0] + " " + w_array[1] + "<br>" + w_array[2];
                }
                ids.push(w_id);
                labels.push(w_label);
                values.push(num_docs);
                parents.push(group_name);
            }
        }
        return {labels: labels, values: values, parents: parents, ids:ids};
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
            "values": values,
            "leaf": {"opacity": 0.7},
            "marker": {"line": {"width": 2}},
            "branchvalues": 'total',
            "outsidetextfont": {size: 24, color: "#377eb8"},
            "insidetextfont": {size: 16},
            'insidetextorientation': "horizontal",
            // Label text orientation
            "textposition": 'inside',
            'hoverinfo': 'label',
        }];

        const layout = {
            "margin": {"l": 10, "r": 10, "b": 10, "t": 10},
            // "sunburstcolorway": D3Colors
        };
        const chart_id = chart_div.attr('id');
        Plotly.newPlot(chart_id, data, layout, {showSendToCloud: true})
        const chart_element = document.getElementById(chart_id);

        // Define the hover event
        chart_element.on('plotly_click', function(data){
            const id = data.points[0].id;
            if(id.startsWith('Topic')){
                const index = parseInt(id.split("#")[1]) - 1;
                const group = group_data[index];
                console.log(group);
                const word_docs = group['word_docIds'];
                const chart_div = $('#lda_topic_network_graph');
                const network_graph = new D3NetworkGraph(word_docs, false, chart_div, D3Colors[index]);
            }else if(id.startsWith('Group')){
                const index = parseInt(id.split("#")[1]) - 1;
                const group = group_data[index];
                console.log(group);
                const word_docs = group['word_docIds'];
                const chart_div = $('#phrase_network_graph');
                const network_graph = new D3NetworkGraph(word_docs, true, chart_div, D3Colors[index]);
            }

        })


    }
    // Create the header
    function create_header(){
        const total_score = group_data.reduce((pre, cur) => cur['score'] + pre, 0);
        const avg_score = total_score / group_data.length;
        const group_count = group_data.length;
        if(is_key_phrase){
            $('#phrase_count').text(group_count);
            $('#phrase_score').text(avg_score.toFixed(2));
            if(avg_score < 0){
                $('#phrase_score').attr('class', 'text-danger');
            }else{
                $('#phrase_score').removeAttr("class");
            }
        }else{
            $('#topic_count').text(group_count);
            $('#topic_score').text(avg_score.toFixed(2));
            if(avg_score < 0){
                $('#topic_score').attr('class', 'text-danger');
            }else{
                $('#topic_score').removeAttr("class");
            }
        }
    }

    // Create the sunburst graph using D3 library
    function createUI() {
        try {
            create_sunburst_graph();
            create_header();
        } catch (error) {
            console.error(error);
        }
    }

    createUI();
}
