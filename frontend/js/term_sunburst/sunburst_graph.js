// Create Plotly sunburst graph to display Grouped key phrases and LDA Topics
// Ref: https://plotly.com/javascript/sunburst-charts/
function SunburstGraph(key_phrase_groups, cluster_no, is_key_phrase) {
    const chart_div = (is_key_phrase?  $('#key_phrase_chart'): $('#lda_topic_chart'));
    // Convert the word_docs map to nodes/links
    const {total, group_total_list} = compute_total(key_phrase_groups);
    // console.log(total);
    // console.log(group_total_list);
    const {labels, values, parents} = create_graph_data();
    console.log(labels);
    console.log(values);
    console.log(parents);
    const width = 600;
    const height = 600;

    // Count the count of a groups
    function compute_total(key_phrase_groups){
        let total = 0;
        let group_total_list = [];
        for(const group of key_phrase_groups){
            let group_total = 0;
            for(const [top_word, doc_ids] of Object.entries(group['word_docIds'])){
                group_total += doc_ids.length;
                total += doc_ids.length;
            }
            group_total_list.push(group_total);
        }
        return {total: total, group_total_list: group_total_list};
    }

    // Convert the json to the format of plotly graph
    function create_graph_data() {
        // Populate the groups of key phrases
        const root = "Cluster#"+cluster_no;
        let labels = [root];
        let values = [total];
        let parents = [""];
        for(let i=0; i < key_phrase_groups.length; i++){
            const prefix = (is_key_phrase ? "Group": "Topic");
            const group_name = prefix + " #" + (i + 1);
            labels.push(group_name);
            values.push(group_total_list[i]);
            parents.push(root);
            // Added the top 10 word
            const group = key_phrase_groups[i];
            const top_words = group['top_words'];
            for(const top_word of top_words){
                const num_docs = group['word_docIds'][top_word].length;
                labels.push(top_word);
                values.push(num_docs);
                parents.push(group_name);
            }

            // console.log(group);
        }
        return {labels: labels, values: values, parents: parents};
    }

    function create_sunburst_graph() {
        chart_div.empty();
        const data = [{
            "type": "sunburst",
            "labels": labels,
            "parents": parents,
            "values": values,
            "leaf": {"opacity": 0.7},
            "marker": {"line": {"width": 2}},
            "branchvalues": 'total'
        }];
        const layout = {
            "margin": {"l": 10, "r": 10, "b": 10, "t": 10},
        };

        Plotly.newPlot(chart_div.attr('id'), data, layout, {showSendToCloud: true})

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
