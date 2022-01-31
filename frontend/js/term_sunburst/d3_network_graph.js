// Create D3 network graph using collocation and
function D3NetworkGraph(word_docs, is_key_phrase, chart_div, color) {
    const chart_id = chart_div.attr('id');
    // Convert the word_docs map to nodes/links
    const {nodes, links} = create_node_link_data(word_docs);
    // console.log(nodes);
    // console.log(links);
    const width = 600;
    const height = 600;
    const max_radius = 30;
    let node_radius = 2;
    let link_width = 1;
    let forceNode = d3.forceManyBody().strength(-2000);
    let forceLink = d3.forceLink(links).id(d => d.id);
    if(is_key_phrase){
        forceNode = d3.forceManyBody();
        forceLink = d3.forceLink(links).id(d => d.id).distance(100);
        node_radius = 5;
        link_width = 5;
    }

    // Convert the json to the format of D3 network graph
    function create_node_link_data(word_docs) {
        // Convert the word_docs to a list
        // Populate the nodes with collocation data
        let nodes = [];
        let id = 0;
        // Add other nodes from term map
        for (const [word, doc_ids] of Object.entries(word_docs) ) {
            const node = {'id': id, 'name': word, 'doc_ids': doc_ids};
            nodes.push(node);
            id += 1;
        }
        // Sort the nodes by the doc_ids
        nodes.sort((a, b) => a['doc_ids'].length - b['doc_ids'].length);
        // console.log(nodes);
        // Populate the links with occurrences
        let links = [];
        for (let source = 0; source < nodes.length; source++) {
            for (let target = source + 1; target < nodes.length; target++) {
                const source_doc_ids = nodes[source]['doc_ids'];
                const target_doc_ids = nodes[target]['doc_ids'];
                const occ_doc_ids = source_doc_ids.filter(doc_id => target_doc_ids.includes(doc_id));
                // console.log(occ_doc_ids);
                if(occ_doc_ids.length > 0){
                    // Add the link
                    links.push({source: nodes[source], target: nodes[target], doc_ids: occ_doc_ids,
                                weight: occ_doc_ids.length});
                }
            }
        }
        return {nodes: nodes, links: links};
    }

    // Get the number of documents for a collocation node
    function get_node_size(node) {
        let num_doc = node['doc_ids'].length;
        // return Math.round(num_doc + max_radius;
        return Math.min(num_doc*node_radius, max_radius);
    }

    // Get the link weight
    function get_link_width(link){
        return link.weight * link_width;
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

    // Create a network graph using D3
    function create_d3_network_graph() {
        // Add the svg node to 'term_map' div
        let svg = d3.select('#'+chart_id).append("svg")
            .attr("width", width)
            .attr("height", height)
            .attr("viewBox", [-width / 2, -height / 2, width, height])
            .attr("style", "max-width: 100%; height: auto; height: intrinsic;");
    //     // Simulation
        const simulation = d3.forceSimulation(nodes)
            .force('link', forceLink)
            .force("charge", forceNode)
            .force('center', d3.forceCenter())
            .on("tick", ticked);

        // Initialise the links
        let link = svg.append("g")
                .attr("class", "link")
                .attr("stroke", "black")
                .attr("stroke-opacity", 0.1)
            .selectAll("line")
            .data(links)
            .join("line")
            .attr("stroke-width", d => get_link_width(d));

        // Initialise the nodes
        let node = svg.append('g')
            .selectAll("g")
            .data(nodes)
            .join("g")
            .call(drag(simulation));

        // Add the circles
        node.append("circle")
            .attr("stroke", "white")
            .attr("stroke-width", 1.5)
            .attr("r", d => get_node_size(d))
            .attr("fill", d => color);
        // node.append("image")
        //     .attr('href', d => get_image_path(d))
        //     .attr('x', d => -1 * get_node_size(d)/2 )
        //     .attr('y', d => -1 * get_node_size(d)/2)
        //     .attr("stroke", "white")
        //     .attr("stroke-width", 1.5)
        //     .attr("width", d => get_node_size(d))
        //     .attr("height", d => get_node_size(d))
        //     .attr('rx', "3")
        //     .attr("color", d => colors(d));

        // Add node label
        node.append("text")
            .attr("class", "lead")
            .attr('x', "1em")
            .attr('y', "0.5em")
            .style("font", d => "14px sans-serif")
            .text(d => {
                return d.name;
            });
        // // Tooltip
        node.append("title").text(d => {
            // Find the source node of d
            const d_links = links.filter(link => link['source'].name === d.name || link['target'].name === d.name);
            // Aggregate all the doc ids to a set to avoid duplicate doc ids
            const occ_doc_ids = new Set();
            for(const d_link of d_links){
                for (const doc_id of d_link['doc_ids']){
                    occ_doc_ids.add(doc_id);
                }
            }
            let str = "'" + d.name + "' alone appears in " + d['doc_ids'].length + " papers\n";
            if(d_links.length > 0){
                str += "and co-occurs with " + d_links.length + " words in " + occ_doc_ids.size + " papers\n";
            }
            return str;
        });

        // Simulate the tick event
        function ticked() {
            link
                .attr("x1", d => d.source.x)
                .attr("y1", d => d.source.y)
                .attr("x2", d => d.target.x)
                .attr("y2", d => d.target.y);

            node.attr("transform", d => `translate(${d.x},${d.y})`);
        }
    }
    //
    //
    // Create the network graph using D3 library
    function _createUI() {
        try {
            chart_div.empty(); // Clear the SVG graph
            create_d3_network_graph();
        } catch (error) {
            console.error(error);
        }
    }

    _createUI();
}
