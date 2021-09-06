// Create D3 network graph
function D3NetworkGraph(collocation_data, occurrence_data){
    const margin = {top: 10, right: 10, left: 10, bottom: 10};
    const width = 600;
    const height = 600;
    const max_radius = 20;
    const {
        node_link_data,
        doc_id_set,
        max_doc_ides,
        max_link_length
    } = Utility.create_node_link_data(collocation_data, occurrence_data);
    const links = node_link_data.links;
    const nodes = node_link_data.nodes;

    // Get the number of documents for a collocation node
    function get_node_size(node_name) {
        let collocation = collocation_data.find(({Collocation}) => Collocation === node_name);
        let col_doc_ids = collocation['DocIDs'];
        let num_doc = 0;
        // Get the values of 'doc_ids'
        for (const year in col_doc_ids) {
            const doc_ids = col_doc_ids[year];
            num_doc += doc_ids.length;
        }
        let radius = num_doc / max_doc_ides * max_radius;
        return Math.max(Math.round(radius), 5);  // Round the radius to the integer
    }

    // Get the number of documents for a link (between two terms
    function get_link_size(link) {
        let source = link.source;
        let target = link.target;
        let occ = occurrence_data['occurrences'][source.id][target.id];
        return Math.sqrt(occ.length);
    }

    // Drag event function
    const drag = function(simulation) {

        function dragstarted(event) {
            if (!event.active) simulation.alphaTarget(0.3).restart();
            event.subject.fx = event.subject.x;
            event.subject.fy = event.subject.y;
        }

        function dragged(event) {
            event.subject.fx = event.x;
            event.subject.fy = event.y;
        }

        function dragended(event) {
            if (!event.active) simulation.alphaTarget(0);
            event.subject.fx = null;
            event.subject.fy = null;
        }

        return d3.drag()
            .on("start", dragstarted)
            .on("drag", dragged)
            .on("end", dragended);
    }
    const colors = function (d) {
        return d3.schemeCategory10[0];
        // let scale = d3.scaleOrdinal(d3.schemeCategory10);
        // return d => scale(d.group);
    }
    // Create the network graph using D3 library
    function _createUI() {
        $('#term_map').empty(); // Clear the SVG graph

        // Simulation
        const simulation = d3.forceSimulation(nodes)
            .force('link', d3.forceLink(links).id(d => d.id).distance(100))
            .force("charge", d3.forceManyBody().strength(-400))
            .force('center', d3.forceCenter(width / 3, height / 3));

        // Add the svg node to 'term_map' div
        const svg = d3.select('#term_map')
            .append("svg").attr("viewBox", [0, 0, width, height])
            .attr('transform', `translate(${margin.left}, ${margin.top})`);
        // Initialise the links
        const link = svg.append('g')
            .attr("stroke", "#999")
            .attr("stroke-opacity", 0.3)
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
            .attr("r", 5)
            .attr("fill", d => colors(d))
            .call(drag(simulation));
        // Add the circles
        // node.append("circle")
        //     .attr('r', d => get_node_size(d.name))
        //     .attr("fill", '#69b3a2');
        // Add the onclick event
        // node.on("click", function (d) {
        //     _create_collocation_document_list_view(d);
        // });
        // Add node label
        node.append("text")
            .attr('x', 8)
            .attr('y', "0.31em")
            .text(d => function(d){
                console.log(d);
                return d.name;
            })
        // Tip tip
        node.append("title")
            .text(d => d.name);

        // Simulate the tick event
        simulation.on('tick', () => {
            link.attr('x1', d => d.source.x)
                .attr('y1', d => d.source.y)
                .attr('x2', d => d.target.x)
                .attr('y2', d => d.target.y);
            node.attr('cx', d => d.x)
                .attr('cy', d => d.y);
        });
    }
    _createUI();
}
