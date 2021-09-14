// Create D3 network graph using collocation and
function D3NetworkGraph(searched_term, term_map, documents) {
    const width = 600;
    const height = 600;
    const max_radius = 10;
    const image_path = 'images/outline_article_black_48dp.png';
    const distance = 120;
    const strength = -1000;
    const {nodes, links} = TermChartUtility.create_node_link_data(searched_term, term_map, documents);
    console.log(nodes);
    console.log(links);
    // Get the color of collocation
    const colors = function (d) {
        return d3.schemeCategory10[d.group];
    }
    // Get font size
    function get_font_size(node){
        if(node.name === searched_term){
            return "20";
        }
        return "14";
    }

    // Get the image_path
    function get_image_path(node){
        // if(node.name !== searched_term){
        //     return 'images/baseline_article_black_48dp.png';
        // }
        return 'images/outline_article_black_48dp.png';
    }

    // Get the number of documents for a collocation node
    function get_node_size(node) {
        let num_doc = node['doc_ids'].length;
        return Math.round(Math.sqrt(num_doc * 100)) + max_radius;
        // return Math.min(num_doc * 10, max_radius);
    }

    // Get the link color
    function get_link_color(link) {
        let source = link.source;
        let target = link.target;
        let source_color = colors(source);
        let target_color = colors(target);
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

    function _create_d3_network_graph(svg) {
        // Simulation
        const simulation = d3.forceSimulation(nodes)
            .force('link', d3.forceLink(links).id(d => d.id).distance(distance))
            .force("charge", d3.forceManyBody().strength(strength))
            .force('center', d3.forceCenter(width / 2, height / 2));

        // Initialise the links
        const link = svg.append('g')
            .attr("stroke-opacity", 0.1)
            .selectAll('line')
            .data(links)
            .join("line")
            .attr("stroke", d => get_link_color(d))
            .attr("stroke-width", d => d.value);

        // Initialise the nodes
        const node = svg.append('g')
            .selectAll("g")
            .data(nodes)
            .join("g")
            .on("click", function (d, n) {// Add the onclick event
                // console.log(n.name);
                let key_term = n.name;
                if (key_term === searched_term) {
                    $('#complementary_term').text("");
                    let doc_list_view = new DocumentListView(searched_term, [], documents);
                    return;
                }
                $('#complementary_term').text(key_term);
                // console.log(complementary_terms);
                const filtered_documents = TermChartUtility.filter_documents_by_key_terms(searched_term,
                    [key_term], term_map, documents);
                console.log(filtered_documents);
                // Create the document list view
                let doc_list_view = new DocumentListView(searched_term, [key_term], filtered_documents);
            }).call(drag(simulation));

        // Add the circles
        // node.append("circle")
        //     .attr("stroke", "white")
        //     .attr("stroke-width", 1.5)
        //     .attr("r", d => get_node_size(d.name))
        //     .attr("fill", d => colors(d));
        node.append("image")
            .attr('href', d => get_image_path(d))
            .attr('x', d => -1 * get_node_size(d)/2 )
            .attr('y', d => -1 * get_node_size(d)/2)
            .attr("stroke", "white")
            .attr("stroke-width", 1.5)
            .attr("width", d => get_node_size(d))
            .attr("height", d => get_node_size(d))
            .attr('rx', "3")
            .attr("color", d => colors(d));


        // Add node label
        node.append("text")
            .attr("class", "lead")
            .attr('x', "0.8em")
            .attr('y', "0.3em")
            .style("font", d => get_font_size(d) + "px sans-serif")
            .text(d => {
                return d.name;
            });
        // Tooltip
        node.append("title")
            .text(d => "'" + d.name + "' has " + term_map.find(tm => tm[0] === d.name)[1].length + " articles");

        // Simulate the tick event
        simulation.on('tick', () => {
            link.attr('x1', d => d.source.x)
                .attr('y1', d => d.source.y)
                .attr('x2', d => d.target.x)
                .attr('y2', d => d.target.y);
            node.attr("transform", d => `translate(${d.x},${d.y})`);
        });
    }


    // Create the network graph using D3 library
    function _createUI() {
        $('#term_chart').empty(); // Clear the SVG graph
        try {
            // Add the svg node to 'term_map' div
            const svg = d3.select('#term_chart')
                .append("svg").attr("viewBox", [0, 0, width, height]);
                // .style("font", font_size + "px sans-serif");
            _create_d3_network_graph(svg);
            let selected_term_view = new SelectedTermView(searched_term, documents);
            let doc_list_view = new DocumentListView(searched_term, [], documents);

        } catch (error) {
            console.error(error);
        }
    }

    _createUI();
}
