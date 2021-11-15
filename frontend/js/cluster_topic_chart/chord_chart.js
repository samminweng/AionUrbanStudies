// Create a chord chart to display the similarity
// Ref: https://www.d3-graph-gallery.com/graph/chord_basic.html
function ChordChart(cluster_sim_matrix, cluster_topics) {
    const width = 600;
    const height = 700;
    const cluster_names = cluster_topics.map(c => "#" + c['cluster_no']);
    // Optimal color pallets for 23 clusters
    // ref: http://vrl.cs.brown.edu/color
    const cluster_colors = ["rgb(104,175,252)", "rgb(79,40,175)", "rgb(153,109,219)", "rgb(47,66,133)", "rgb(62,234,239)",
        "rgb(37,115,139)", "rgb(179,228,103)", "rgb(39,122,53)", "rgb(103,240,89)", "rgb(117,72,25)",
        "rgb(252,206,106)", "rgb(179,65,108)", "rgb(196,145,150)", "rgb(192,0,24)", "rgb(254,133,173)",
        "rgb(248,35,135)", "rgb(254,143,6)", "rgb(169,190,175)", "rgb(178,139,40)", "rgb(239,102,240)",
        "#1e90ff", "#db7093", "#b0e0e6",];

    function createChordChart() {
        let svg = d3.select("#cluster_topic_chart").append("svg")
            .attr("width", width)
            .attr("height", height)
            .attr("viewBox", [-width / 2, -height / 2, width, height])      // Set the view box to see the chart
            .append("g");

        const chord_data = d3.chord()
            .padAngle(0.05)     // Padding between clusters
            (cluster_sim_matrix);
        console.log(chord_data);

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

        // Add text to the path to display the cluster name
        paths.selectAll("text")
            .data(d => d.groups)
            .enter()
            .append("text")
            .attr('transform', d => `translate(${arc.centroid(d)})`)    // Place the center of the path
            .attr('text-anchor', 'middle')
            .attr('font', 'bold 20px sans-serif')
            .text(d => "#" + d.index)

        // Add links to groups
        const ribbon = d3.ribbon().radius(200);
        const links = svg.datum(chord_data).append('g')
        links.selectAll('path')
            .data(d => d)
            .join('path')
            .attr('d', ribbon)
            .style("fill", d => cluster_colors[d.source.index] )
            .on('mouseover', )

        // .style("stroke", "black");


    }


    function _createUI() {
        // Clear the chart view and Set width and height
        $('#cluster_topic_chart').empty();
        $('#cluster_topic_chart').width(width).height(height);

        createChordChart();
    }

    _createUI();
}
