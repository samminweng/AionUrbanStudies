function NetworkChart(_corpus_data, _collocation_data, _occurrence_data) {
    const corpus_data = _corpus_data;
    const collocation_data = _collocation_data; // Describe the terms
    const occurrence_data = _occurrence_data; // Describe the number of document ids between two terms
    const margin = {top: 10, right: 10, left: 10, bottom: 10};
    const width = 600;
    const height = 600;
    const max_radius = 20;
    const {node_link_data, doc_id_set, max_doc_ides, max_link_length} = Utility.create_node_link_data(collocation_data, occurrence_data);
    const links = node_link_data.links.map(d => Object.create(d));
    const nodes = node_link_data.nodes.map(d => Object.create(d));

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
        let radius = num_doc / max_doc_ides * max_radius;
        return Math.round(radius);  // Round the radius to the integer
    }

    // Get the number of documents for a link (between two terms
    function get_link_size(link){
        let source = link.source;
        let target = link.target;
        let occ = occurrence_data['occurrences'][source][target];
        return Math.sqrt(occ.length);
    }

    // Create the network graph using D3 library
    function _create_d3_network_chart() {
        // Add the svg node to 'term_map' div
        const svg = d3.select('#term_map')
                      .append("svg").attr("viewBox", [0, 0, width, height])
                      .attr('transform', `translate(${margin.left}, ${margin.top})`);
        // Initialise the links
        const link = svg.append('g')
            .attr("stroke", "#999")
            .attr("stroke-opacity", 0.6)
            .selectAll('line')
            .data(links)
            .join("line")
            .attr("stroke-width", d => get_link_size(d));
        // Initialise the nodes
        const node = svg.append('g')
            .attr("stroke", "#fff")
            .attr("stroke-width", 1.5)
            .selectAll("circle")
                        .data(nodes)
                        .join("circle")
                        .attr('r', d => get_node_size(d.name))
                        .attr("fill", '#69b3a2');
        node.append("title")
            .text(d => d.name);

        // Simulation
        const simulation = d3.forceSimulation(nodes)
                             .force('link', d3.forceLink(links).id(d => d.id))
                             .force("charge", d3.forceManyBody().strength(-1000))
                             .force('center', d3.forceCenter(width/3, height/3))
                             .on('end', ticked);

        function ticked(){
            link.attr('x1', d => d.source.x)
                .attr('y1', d => d.source.y)
                .attr('x2', d => d.target.x)
                .attr('y2', d => d.target.y);
            node.attr('cx', d => d.x)
                .attr('cy', d => d.y);
        }

    }

    // Create a document list view for a collocation
    function _create_collocation_document_list_view(){
        // For testing, we create a document list view for 'machine learning'
        let collocation = collocation_data.find(({index}) => index === 0);
        let documents = Utility.collect_documents(collocation, corpus_data);
        console.log(documents);
        let doc_list_view = new DocumentListView(documents);

    }


    function _createUI() {
        _create_d3_network_chart(); // Create the nodes and links of term occurrences.
        _create_collocation_document_list_view();

    }

    _createUI();
}
