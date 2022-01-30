// Create Plotly sunburst graph to display Grouped key phrases and LDA Topics
// Ref: https://plotly.com/javascript/sunburst-charts/
function SunburstGraph(key_phrase_groups, cluster_no, cluster_docs) {
    $('#doc_list').empty();
    $('#header').empty();
    // Convert the word_docs map to nodes/links
    const {labels, values} = create_node_data();
    // console.log(nodes);
    // console.log(links);
    const width = 600;
    const height = 600;
    const max_radius = 30;


    // Convert the json to the format of plotly graph
    function create_node_data() {
        // Populate the groups of key phrases
        let labels = ["cluster#"];
        let values = [];
        let group_id = 1;
        for(const group of key_phrase_groups){
            labels.push("Group #" + group_id);
            const num_docs = group['NumDocs'];
            values.push(num_docs);
            // Added the top 10 word
            const key_phrases = group['key-phrases'];
            for(const key_phrase of key_phrases){

            }

            console.log(group);
        }
        return {labels: labels, values: values};
    }

    // Onclick event of a node
    function click_node(node){
        const doc_ids = node.doc_ids;
        const node_docs = cluster_docs.filter(d => doc_ids.includes(d['DocId']));
        // console.log(node_docs);
        const select_term = [node.name];
        // Find the source node of d
        const d_links = links.filter(link => link['source'].name === node.name || link['target'].name === node.name);
        // Aggregate all the doc ids to a set to avoid duplicate doc ids
        const occ_doc_ids = new Set();
        const occ_terms = new Set();
        for(const d_link of d_links){
            for (const doc_id of d_link['doc_ids']){
                occ_doc_ids.add(doc_id);
            }
            occ_terms.add(d_link['source'].name);
            occ_terms.add(d_link['target'].name);
        }
        const occ_term_list = Array.from(occ_terms).filter(term => term !== node.name);
        $("#header").empty();
        const header_div = $("<div class='h5'></div>");
        header_div.append($("<span class='search_terms'>" + node.name  + "</span> " + "<span> appears in " +
                            node_docs.length + " papers</span>"));
        if (occ_term_list.length > 0){
            header_div.append($('<span> and occurs with </span>'));
        }

        // Populate each co-occurring term
        const occ_div = $('<div></div>')
        for(const occ_term of occ_term_list){
            const link_btn = $('<button type="button" class="btn btn-link">' + occ_term+ ' </button>');
            link_btn.button();
            link_btn.click(function( event ) {
                const selected_node = nodes.find(n => n.name === node.name);
                const occ_node = nodes.find(n => n.name === occ_term);
                const occ_doc_ids = selected_node['doc_ids'].filter(doc_id => occ_node['doc_ids'].includes(doc_id));
                const occ_docs = cluster_docs.filter(d => occ_doc_ids.includes(d['DocId']));
                const doc_list = new DocList(occ_docs, select_term, [occ_term]);
                header_div.empty();
                header_div.html("<span class='search_terms'>" + node.name  + "</span> and <mark>" + occ_term +
                                "</mark> both appear in " + occ_docs.length + ' papers')
            });

            occ_div.append(link_btn);
        }
        $("#header").append(header_div);
        $("#header").append(occ_div);
        const doc_list = new DocList(node_docs, select_term, occ_term_list);
    }

    function create_sunburst_graph() {

    }


    // Create the sunburst graph using D3 library
    function createUI() {
        $('#term_chart').empty(); // Clear the SVG graph
        try {
            create_sunburst_graph();
        } catch (error) {
            console.error(error);
        }
    }

    createUI();
}
