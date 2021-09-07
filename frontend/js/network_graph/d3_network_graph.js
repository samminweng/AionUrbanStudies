// Create D3 network graph
function D3NetworkGraph(collocation_data, occurrence_data, corpus_data) {
    // const margin = {top: 10, right: 10, left: 10, bottom: 10};
    const width = 600;
    const height = 600;
    const max_radius = 30;
    const distance = 200;
    const strength = -400;
    const {node_link_data, max_doc_ids} = Utility.create_node_link_data(collocation_data, occurrence_data);
    const links = node_link_data.links;
    const nodes = node_link_data.nodes;
    // Get the color of collocation
    const colors = function (collocation) {
        let group = Utility.get_group_number(collocation)
        return d3.schemeCategory10[group];
        // let scale = d3.scaleOrdinal(d3.schemeCategory10);
        // return d => scale(d.group);
    }

    // Get the number of documents for a collocation node
    function get_node_size(node_name) {
        let num_doc = Utility.get_number_of_documents(node_name, collocation_data);
        let radius = Math.sqrt(num_doc);
        // let radius = num_doc / max_doc_ids * max_radius;
        return Math.round(radius);  // Round the radius to the integer
    }

    // Get the number of documents for a link (between two terms
    function get_link_size(link) {
        let source = link.source;
        let target = link.target;
        let occ = occurrence_data['occurrences'][source.id][target.id];
        return Math.max(1.5, Math.sqrt(occ.length));
    }

    // Get the link color
    function get_link_color(link) {
        let source = link.source;
        let target = link.target;
        let source_color = colors(source.name);
        let target_color = colors(target.name);
        if (source_color !== target_color) {
            // Scale the color
            return d3.schemeCategory10[7];
        }
        return source_color;
    }

    // Drag event function
    const drag = function (simulation) {

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


    // Create the network graph using D3 library
    function _createUI() {
        $('#term_map').empty(); // Clear the SVG graph

        // Simulation
        const simulation = d3.forceSimulation(nodes)
            .force('link', d3.forceLink(links).id(d => d.id).distance(distance))
            .force("charge", d3.forceManyBody().strength(strength))
            .force('center', d3.forceCenter(width / 2, height / 2));

        // Add the svg node to 'term_map' div
        const svg = d3.select('#term_map')
            .append("svg").attr("viewBox", [0, 0, width, height])
            .style("font", "16px sans-serif");

        // Initialise the links
        const link = svg.append('g')
            .selectAll('line')
            .data(links)
            .join("line")
            .attr("stroke", d => get_link_color(d))
            .attr("stroke-width", d => get_link_size(d))
            .attr("stroke-opacity", 0.1);

        // Initialise the nodes
        const node = svg.append('g')
            .attr("stroke-width", 1.5)
            .selectAll("g")
            .data(nodes)
            .join("g")
            .on("click", function (d, n) {// Add the onclick event
                console.log(n.name);
                let key_term = n.name;
                // Check if the selected item
                if (!$('#selected_term_1').is(':empty') && !$('#selected_term_2').is(':empty')) {
                    alert("Please clear the terms");
                    return;
                }
                // Update the selected_term_1
                if ($('#selected_term_1').is(':empty')) {
                    const group_1 = Utility.get_group_number(key_term);
                    $('#selected_term_1')
                        .attr('class', 'keyword-group-' + group_1)
                        .text(key_term);
                    return;
                }
                // Update the selected_term_2
                const group_2 = Utility.get_group_number(key_term);
                $('#selected_term_2').attr('class', 'keyword-group-' + group_2).text(key_term);
                // Get the key term 1 and key term 2
                let collocation_1 = $('#selected_term_1').text();
                let collocation_2 = $('#selected_term_2').text();
                let key_terms = [collocation_1, collocation_2]
                let doc_list_view = new DocumentListView(key_terms, collocation_data, corpus_data);
            })
            .call(drag(simulation));

        // Add the circles
        node.append("circle")
            .attr("stroke", "#aaa")
            .attr("stroke-width", 1.5)
            .attr("r", d => get_node_size(d.name))
            .attr("fill", d => colors(d.name));

        // Add node label
        node.append("text")
            .attr("class", "lead")
            .attr('x', 8)
            .attr('y', "0.31em")
            .text(d => {
                return d.name;
            });
        // Tooltip
        node.append("title")
            .text(d => "'" + d.name + "' has " + Utility.get_number_of_documents(d.name, collocation_data) + " articles");

        // Simulate the tick event
        simulation.on('tick', () => {
            link.attr('x1', d => d.source.x)
                .attr('y1', d => d.source.y)
                .attr('x2', d => d.target.x)
                .attr('y2', d => d.target.y);
            node.attr("transform", d => `translate(${d.x},${d.y})`);
        });
    }

    _createUI();
}
