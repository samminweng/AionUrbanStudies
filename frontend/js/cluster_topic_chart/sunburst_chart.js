// Create a sunburst chart
// Ref: https://plotly.com/javascript/reference/sunburst/
function SunburstChart(cluster_groups, cluster_data, doc_data){
    const width = 600;
    const height = 700;
    console.log(doc_data);
    // Convert the cluster groups, cluster and topics to table
    function convert_to_data_table(){
        // Data table used to plot the sunburst chart
        const total = cluster_data.reduce((pre, cur) => pre + cur['NumDocs'], 0);
        // Add root node ('corpus')
        let data_table =[{
            type: "sunburst",
            labels: ["Corpus"],     // Node name
            parents: [""],      // Parent node name
            values: [total],      // Node size
            text: ['Corpus'],       // Hover text
            customdata: [null],     // Extra list to store the extra data of each node
            textinfo:'label',       // Display text when hovering over a node
            branchvalues: 'total',
            // leaf: {opacity: 0.9},
            textposition: 'inside',
            insidetextorientation: 'radial',
            hoverinfo: "text"
        }];
        // Go through each group
        const g_parent = "Corpus";
        for(const group of cluster_groups){
            const group_name = group['group'];
            const group_clusters = cluster_data.filter(c => group['clusters'].includes(c['Cluster']));
            const group_value = group_clusters.reduce((pre, cur) => pre + cur['NumDocs'], 0);
            const group_percent = 100 * (group_value /total);
            // Add group nodes
            data_table[0]['labels'].push(group_name);
            data_table[0]['parents'].push(g_parent);
            data_table[0]['values'].push(group_value);
            data_table[0]['text'].push(group_name + " has <b>" + group_percent.toFixed() + "% of articles</b>");
            data_table[0]['customdata'].push(group);
            // Go through each cluster within a group
            for(const cluster of group_clusters){
                // console.log(cluster);
                const cluster_no = cluster['Cluster'];
                const cluster_name = "Cluster #" + cluster_no;
                const cluster_docs = doc_data.filter(d => cluster['DocIds'].includes(d['DocId']));
                const cluster_percent = 100 * (cluster_docs.length / total);
                const cluster_topics = cluster['TopicN-gram'].slice(0, 10); // Get top 10 topics
                const cluster_text = cluster_name + ' has <b>' + cluster_percent.toFixed()+ '% of articles</b><br><br>'
                    + cluster_topics.map(t => '<b>' + t['topic'] + '</b>').join("<br>");
                // Add cluster nodes
                data_table[0]['labels'].push(cluster_name);
                data_table[0]['parents'].push(group_name);
                data_table[0]['values'].push(cluster_docs.length);
                data_table[0]['text'].push(cluster_text);
                data_table[0]['customdata'].push(cluster);
            }
        }
        return data_table;
    }



    function _createUI(){
        // Set width and height
        $('#cluster_topic_chart').empty();
        $('#cluster_topic_chart').width(width).height(height);

        // Draw sunburst chart
        const data_table = convert_to_data_table();
        const options = {
            margin: {l: 30, r: 30, b: 30, t: 30},
            width: width,
            height: height,
            hovermode: "closest",
            hoverlabel: { bgcolor: "#FFF" },
            config: { responsive: true }

        };
        console.log(data_table);
        // Draw the sunburst chart
        Plotly.newPlot('cluster_topic_chart', data_table, options);

        const chart = document.getElementById('cluster_topic_chart');
        // Define Onclick event of the chart
        chart.on('plotly_click', function(data){
            // console.log(data);
            if(data.points.length> 0 && data.points[0].customdata !== null){
                const custom_data = data.points[0].customdata;
                if('Cluster' in custom_data){
                    const cluster = custom_data;
                    const cluster_no = cluster['Cluster'];
                    const cluster_docs = doc_data.filter(d => cluster['DocIds'].includes(d['DocId']));
                    const cluster_topics = cluster['TopicN-gram'].slice(0, 30); // Get top 30 topics
                    // Update the heading
                    $('#topic_doc_heading').text("Cluster #" + cluster_no + " has " + cluster_docs.length + " articles.");
                    const topic_list = new TopicList(cluster_topics, cluster_docs);
                    const doc_list = new DocList(cluster_docs, cluster_topics, null);
                    // console.log("custom_data = " +  custom_data);
                }
            }
        });

    }


    _createUI();
}
