// Create D3 network graph using collocation and
function D3NetworkGraph(topic_data, cluster_docs) {
    const width = 600;
    const height = 600;
    const max_radius = 30;
    const distance = 200;
    // const strength = -1000;
    const word_docs = topic_data['word_docIds'];
    // Convert the word_docs map to nodes/links
    const {nodes, links} = create_node_link_data(word_docs);
    console.log(nodes);
    console.log(links);
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
        console.log(nodes);
        // Populate the links with occurrences
        let links = [];
        for (let source = 0; source < nodes.length; source++) {
            for (let target = source + 1; target < nodes.length; target++) {
                const source_doc_ids = nodes[source]['doc_ids'];
                const target_doc_ids = nodes[target]['doc_ids'];
                const occ_doc_ids = source_doc_ids.filter(doc_id => target_doc_ids.includes(doc_id));
                console.log(occ_doc_ids);
                if(occ_doc_ids.length > 0){
                    // Add the link
                    links.push({source: nodes[source], target: nodes[target], value: occ_doc_ids, weight: occ_doc_ids.length});
                }
            }
        }
        return {nodes: nodes, links: links};
    }


    // Get the color of collocation
    const colors = function (d) {
        return d3.schemeCategory10[0];
    }
    // Get font size
    function get_font_size(node){
        return "12";
    }

    // Get the number of documents for a collocation node
    function get_node_size(node) {
        let num_doc = node['doc_ids'].length;
        // return Math.round(num_doc + max_radius;
        return Math.min(num_doc * 2, max_radius);
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

    function _create_d3_network_graph() {
        // Add the svg node to 'term_map' div
        let svg = d3.select('#term_chart')
            .append("svg").attr("viewBox", [0, 0, width, height]);
    //     // Simulation
        const simulation = d3.forceSimulation(nodes)
            // .force('link', d3.forceLink(links).id(d => d.id).distance(distance))
            .force("charge", d3.forceManyBody().strength(-10))
            .force('center', d3.forceCenter(width / 2, height / 2))
            .on("tick", ticked);

        // Initialise the links
        let link = svg.append("g")
                .attr("class", "link")
                .attr("stroke", "black")
                .attr("stroke-opacity", 0.1)
            .selectAll("line")
            .data(links)
            .join("line")
            .attr("stroke-width", function(d){
                // console.log(d);
                return d.weight;
            }) ;

        // Initialise the nodes
        let node = svg.append('g')
            .selectAll("g")
            .data(nodes)
            .join("g")
            .on("click", function (d, n) {// Add the onclick event
                // // console.log(n.name);
                // let key_term = n.name;
                // if (key_term === searched_term) {
                //     $('#complementary_term').text("");
                //     let doc_list_view = new DocumentListView(searched_term, [], documents);
                //     return;
                // }
                // $('#complementary_term').text(key_term);
                // // console.log(complementary_terms);
                // const filtered_documents = TermChartUtility.filter_documents_by_key_terms(searched_term,
                //     [key_term], term_map, documents);
                // console.log(filtered_documents);
                // // Create the document list view
                // let doc_list_view = new DocumentListView(searched_term, [key_term], filtered_documents);
            }).call(drag(simulation));

        // Add the circles
        node.append("circle")
            .attr("stroke", "white")
            .attr("stroke-width", 1.5)
            .attr("r", d => get_node_size(d))
            .attr("fill", d => d3.schemeCategory10[0]);
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
            .style("font", d => get_font_size(d) + "px sans-serif")
            .text(d => {
                return d.name;
            });
        // // Tooltip
        // node.append("title")
        //     .text(d => "'" + d.name + "' has " + term_map.find(tm => tm[0] === d.name)[1].length + " articles");
        // link = svg.selectAll('.link');
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
        $('#term_chart').empty(); // Clear the SVG graph
        try {
            _create_d3_network_graph();
            // let selected_term_view = new SelectedTermView(searched_term, documents);
            // let doc_list_view = new DocumentListView(searched_term, [], documents);

        } catch (error) {
            console.error(error);
        }
    }

    _createUI();
}
