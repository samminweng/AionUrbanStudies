// Create a chord chart to display the similarity
// Ref: https://www.d3-graph-gallery.com/graph/chord_basic.html
function ChordChart(cluster_sim_data, cluster_topics) {
    const width = 600;
    const height = 700;
    // const cluster_names = cluster_topics.map(c => "#" + c['cluster_no']);
    // Optimal color pallets for 23 clusters
    // ref: http://vrl.cs.brown.edu/color
    const cluster_colors = ["rgb(104,175,252)", "rgb(79,40,175)", "rgb(153,109,219)", "rgb(47,66,133)", "rgb(62,234,239)",
        "rgb(37,115,139)", "rgb(179,228,103)", "rgb(39,122,53)", "rgb(103,240,89)", "rgb(117,72,25)",
        "rgb(252,206,106)", "rgb(179,65,108)", "rgb(196,145,150)", "rgb(192,0,24)", "rgb(254,133,173)",
        "rgb(248,35,135)", "rgb(254,143,6)", "rgb(169,190,175)", "rgb(178,139,40)", "rgb(239,102,240)",
        "#1e90ff", "#db7093", "#b0e0e6",];
    // Convert similarity matrix into square matrix
    const cluster_sim_matrix = convert_to_square_matrix(cluster_sim_data);

    // Create a tooltip div
    const tooltip = d3.select("body")
        .append("div")
        .style("max-width", width/5)
        .style("position", "absolute")
        .style("z-index", "10")
        .style("visibility", "hidden")
        .style("background", "white");

    // .text("a simple tooltip");
    // Convert the similarity matrix as square matrix
    function convert_to_square_matrix(cluster_sim_data) {
        const square_matrix = [];
        for (let i = 0; i < cluster_sim_data.length; i++) {
            let row = []
            for (let j = 0; j < cluster_sim_data.length; j++) {
                if (i !== j) {
                    const sim = cluster_sim_data[i][j];
                    const percent = Math.round(sim * 100);
                    row.push(percent);
                } else {
                    row.push(0);        // The same cluster has no similarity
                }

            }
            square_matrix.push(row);
        }
        // console.log(square_matrix);
        return square_matrix;
    }

    // Reset opacity
    function reset_links(svg){
        // Reset the links to 1 (show all the links)
        svg.selectAll("path.link")
            .transition()
            .style("opacity", 1);
    }

    // Display the relevant link
    function highlight_links(svg, cluster_no){
        svg.selectAll("path.link")
            .transition()
            .style("opacity", 0);
        svg.selectAll("path.link")
            .filter(function (link) {
                return link.source.index === cluster_no || link.target.index === cluster_no;
            })
            .transition()
            .style("opacity", 1);
    }

    // Get the cluster topic and similarity
    function show_tooltip(top, left, src, target, sim){
        try{
            // Set tooltip position
            tooltip.style("visibility", "visible");
            tooltip.style("top", top+"px")
                .style("left", left+"px");
            const tooltip_div = $('<div class="small"></div>');
            tooltip_div.append($("<div class='h5'>" + sim + '% of similarity between Cluster #' + src + ' and #' + target + "</div>"))
            // Get top 10 topics of source cluster
            const src_cluster_topics = cluster_topics.find(c => c['cluster_no'] === src)['topics'].join(", ");
            tooltip_div.append($("<div class='p3'>Cluster #" + src+ "  Top 10 topics</div>")
                                .append($("<div class='small'>" + src_cluster_topics+"</div>")));
            // Get top 10 topics of target cluster
            const target_cluster_topics =  cluster_topics.find(c => c['cluster_no'] === target)['topics'].join(", ");
            tooltip_div.append($("<div class='p3'> Cluster #" + target+ " Top 10 topics</div>")
                .append($("<div class='small'>" + target_cluster_topics+"</div>")));

            tooltip.html(tooltip_div.html());
        }catch (error) {
            console.error(error);
        }

        // Set top and left position of tooltip
        // tooltip.css({top: (e.pageY-10)+"px" , left: (e.pageX+10) + "px", position:'absolute'});
    }


    function createChordChart() {
        try {
            const svg = d3.select("#cluster_topic_chart").append("svg")
                .attr("width", width)
                .attr("height", height)
                .attr("viewBox", [-width / 2, -height / 2, width, height])      // Set the view box to see the chart
                .append("g");

            const chord_data = d3.chord()
                .padAngle(0.05)     // Padding between clusters
                (cluster_sim_matrix);
            // console.log(chord_data);

            // Add groups to inner circles
            const paths = svg.datum(chord_data).append("g");
            const arc = d3.arc().innerRadius(200).outerRadius(235);     // Inner circle radius and outer circle radius
            // Add a color path to represent the cluster
            paths.selectAll("path")
                .data(d => d.groups)
                .enter()
                .append("path")
                .style("fill", d => cluster_colors[d.index])    // Cluster id is the index of color palette
                .attr("d", arc)
                .on('mouseover', (e, d) => highlight_links(svg, d.index))        // Change opacity of all links to zero (hide all the links)
                .on('mouseout', d => reset_links(svg));


            // Add text to the path to display the cluster name
            paths.selectAll("text")
                .data(d => d.groups)
                .enter()
                .append("text")
                .attr('transform', d => `translate(${arc.centroid(d)})`)    // Place the center of the path
                .attr('text-anchor', 'middle')
                .attr('font', 'bold 20px sans-serif')
                .text(d => "#" + d.index)
                .on('mouseover', (e, d) => highlight_links(svg, d.index))
                .on('mouseout', d => reset_links(svg));

            // Add links to groups
            const links = svg.datum(chord_data).append('g');
            links.selectAll('path')
                .data(d => d)
                .join('path')
                .attr('d', d3.ribbon().radius(200))
                .attr("class", "link")
                .style("fill", d => cluster_colors[d.source.index])
                .on('mouseover', function (e, d) {
                    const top = e.pageY-10;
                    const left = e.pageX+10;
                    // console.log(d);
                    // Change opacity of all links to zero (hide all the links)
                    svg.selectAll("path.link")
                        .transition()
                        .style("opacity", 0);
                    // Show all the links outwards the source node
                    const src = d.source.index;
                    const target = d.target.index;
                    const sim = d.source.value;
                    // Make all the link opacity lower
                    svg.selectAll("path.link")
                        .filter(function (link) {
                            return link.source.index === src || link.target.index === src;
                        })
                        .transition()
                        .style("opacity", 0.2);
                    // Make the link opacity stronger
                    svg.selectAll('path.link')
                        .filter(function (link) {
                            return link.source.index === src && link.target.index === target;
                        })
                        .transition()
                        .style("opacity", 1.0);
                    show_tooltip(top, left, src, target, sim);
                })
                .on('mouseout', d => {
                    reset_links(svg);
                    tooltip.style("visibility", "hidden");
                });
                // .append("title")    // Add tooltip
                // .text(function(d){
                //     const src = d.source.index;
                //     const target = d.target.index;
                //     const sim = d.source.value;
                //     // console.log(d);
                //     return create_tooltip(src, target, sim);
                // });
                // Create a custom tooltip div



        } catch (error) {
            console.error(error);
        }
    }


    function _createUI() {
        // Clear the chart view and Set width and height
        $('#cluster_topic_chart').empty();
        $('#cluster_topic_chart').width(width).height(height);
        createChordChart();
    }

    _createUI();
}
