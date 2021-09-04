function NetworkChart(_collocation_data, _occurrence_data) {
    const collocation_data = _collocation_data; // Describe the terms
    const occurrence_data = _occurrence_data; // Describe the number of document ids between two terms
    const margin = {top: 10, right: 10, left: 10, bottom: 10};
    const width = 400 - margin.left - margin.right;
    const height = 400 - margin.top - margin.bottom;
    const {node_link_data, doc_id_set, max_doc_ides} = _create_node_link_data();

    // Convert the collocations
    function _create_node_link_data(){
        let doc_id_set = new Set(); // Store all the document ids
        let max_doc_ides = 0;
        // Populate the nodes with collocation data
        let nodes = [];
        for(let collocation of collocation_data){
            let node = {'id': collocation['index'], 'name': collocation['Collocation']}
            nodes.push(node);
            // Add the doc_ids to doc_id_sets
            let col_doc_ids = collocation['DocIDs'];
            let total_doc_ids = 0;
            for(const year in col_doc_ids){
                const doc_ids = col_doc_ids[year];
                for(const doc_id in doc_ids){
                    doc_id_set.add(doc_id);
                }
                total_doc_ids += doc_ids.length;
            }
            if (max_doc_ides < total_doc_ids){
                max_doc_ides = total_doc_ids;
            }

        }
        console.log(doc_id_set);

        // Populate the links with occurrences
        const occurrences = occurrence_data['occurrences'];
        let links = [];
        for (let source=0; source < nodes.length; source++) {
            for (let target =0; target < nodes.length; target++){
                if (source !== target){
                    let occ = occurrences[source][target];
                    if(occ.length > 0){
                        // Add the link
                        links.push({source: source, target: target});
                    }
                }
            }
        }
        return {node_link_data: {nodes: nodes, links: links}, doc_id_set: doc_id_set, max_doc_ides: max_doc_ides};
    }

    // Get the number of documents for a collocation node
    function get_node_size(node_name){
        let collocation = collocation_data.find(({Collocation}) => Collocation === node_name);
        let col_doc_ids = collocation['DocIDs'];
        let num_doc = 0;
        // Get the values of 'doc_ids'
        for(const year in col_doc_ids){
            const doc_ids = col_doc_ids[year];
            num_doc += doc_ids.length;
        }
        let radius = num_doc / max_doc_ides * 20;
        return Math.round(radius);  // Round the radius to the integer
    }

    // Create the network graph using D3 library
    function _create_d3_network_chart() {
        // Add the svg node to 'term_map' div
        const svg = d3.select('#term_map')
                        .append("svg")
                        .attr('width', 400)
                        .attr('height', 400)
                        .append('g')
                        .attr('transform', 'translate(' + margin.left + ',' + margin.top + ')');
        // Initialise the links
        const link = svg.selectAll('line')
            .data(node_link_data.links)
            .join("line")
            .style("stroke", '#aaa');
        // Initialise the nodes
        const node = svg.selectAll("circle")
                        .data(node_link_data.nodes)
                        .join("circle")
                        .attr('r', d => get_node_size(d.name))
                        .attr("fill", '#69b3a2');

        // Simulation
        const simulation = d3.forceSimulation(node_link_data.nodes)
                             .force('link', d3.forceLink()
                                              .id(d => d.id)
                                              .links(node_link_data.links)
                             )
                            .force("charge", d3.forceManyBody().strength(-800))
            .force('center', d3.forceCenter(width/2, height/2))
            .on('end', ticked);

        function ticked(){
            link.attr('x1', d => d.source.x)
                .attr('y1', d => d.source.y)
                .attr('x2', d => d.target.x)
                .attr('y2', d => d.target.y);
            node.attr('cx', d => d.x + 6)
                .attr('cy', d => d.y - 6);
        }

    }

    function _createUI() {
        _create_d3_network_chart(); // Create the nodes and links of term occurrences.

    }

    _createUI();
}
