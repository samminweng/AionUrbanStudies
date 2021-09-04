function NetworkChart(_collocation_data, _occurrence_data) {
    const collocation_data = _collocation_data; // Describe the terms
    const occurrence_data = _occurrence_data; // Describe the number of document ids between two terms
    const margin = {top: 10, right: 10, left: 10, bottom: 10};
    const width = 400 - margin.left - margin.right;
    const height = 400 - margin.top - margin.bottom;
    const node_link_data = _create_node_link_data();
    // Convert the collocations
    function _create_node_link_data(){
        // Populate the nodes with collocation data
        let nodes = [];
        for(let collocation of collocation_data){
            let node = {'id': collocation['index'], 'name': collocation['Collocation']}
            nodes.push(node);
        }
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
        return {'nodes': nodes, 'links': links};
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
                        .attr('r', 20)
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
            node.attr('cx', d => d.x + 10)
                .attr('cy', d => d.y - 10);
        }

    }

    function _createUI() {
        _create_d3_network_chart(); // Create the nodes and links of term occurrences.

    }

    _createUI();
}
