// Create D3 network graph using collocation and
function D3NetworkGraph(word_docs, cluster_docs, is_key_phrase) {
    $('#doc_list').empty();
    $('#header').empty();
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
        let svg = d3.select('#term_chart').append("svg")
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
            .on("click", function (d, n) {// Add the onclick event
                click_node(n);
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
            return "'" + d.name + "' alone appears in " + d['doc_ids'].length + " papers and\n" +
                        "co-occurs with " + d_links.length + " words in " + occ_doc_ids.size + " papers\n";
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
