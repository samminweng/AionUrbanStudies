function WordTree(cluster_topic_data, doc_term_data){
    const width = 400;
    const height = 600;
    const total_num_docs = doc_term_data.length;
    console.log(cluster_topic_data);

    // Convert cluster topic to data table for Google chart
    function convert_cluster_topics_data_table(){
        let data_table = new google.visualization.DataTable();
        data_table.addColumn('number', 'id');
        data_table.addColumn('string', 'childLabel');
        data_table.addColumn('number', 'parent');
        // data_table.addColumn('number', 'size');
        data_table.addColumn('number', 'weight');
        data_table.addColumn({ role: 'style' });    // Word color
        // data_table.addColumn({type: 'string', role: 'tooltip'});

        let rows = [
            [0, 'Corpus', -1,  total_num_docs, 'black']
        ];
        const root_id = 0;
        let id = 0;
        // Go through each cluster
        for(const cluster of cluster_topic_data){
            if(cluster['Cluster'] >= 0) {
                const c_id = ++id;
                const c_total = cluster['DocIds'].length;
                const cluster_topics = cluster['TopicN-gram'].slice(0, 10);    // Get top 10 topics
                cluster_topics.sort((a, b) => b['doc_ids'].length - a['doc_ids'].length);   // Sort cluster topics by number of doc
                const c_name = 'Cluster #' + cluster['Cluster'];
                const cluster_node = [c_id, c_name, root_id, c_total, 'blue'];
                rows.push(cluster_node);
                // Get the sum of topic document counts
                const sub_total = cluster_topics.reduce((pre, cur) => pre + cur['doc_ids'].length, 0);
                for(const c_topic of cluster_topics){
                    const t_id = ++id;
                    let proportion = c_topic['doc_ids'].length / sub_total * c_total;
                    const t_name = c_topic['topic'];
                    const topic_node = [t_id, c_topic['topic'], c_id, proportion, 'green'];
                    rows.push(topic_node);
                }
            }
        }
        // Add all rows
        data_table.addRows(rows);
        return data_table;
    }




    // Draw the chart
    function drawChart(){
        const data_table = convert_cluster_topics_data_table();
        console.log(data_table);    // Store the relation between clusters and topics
        // const nodeListData = new google.visualization.arrayToDataTable(data_table);
        // Create the word tree
        const wordtree = new google.visualization.WordTree(
            document.getElementById('cluster_topic_chart'));
        // Draw the chart
        wordtree.draw(data_table, {
            maxFontSize: 18,
            wordtree: {
                width: width,
                format: 'explicit',
                type: 'suffix',
                forceIFrame: true,
            },
            tooltip: { trigger: 'selection' }
        });

        // Word tree select event
        function selectHandler(){
            const selectedItem = wordtree.getSelection()[0];
            console.log(selectedItem);
        }



        // Add the select handler
        google.visualization.events.addListener(wordtree, 'select', selectHandler);

    }

    function _createUI(){
        // Set width and height
        $('#cluster_topic_chart').width(width).height(height);

        google.charts.load('current', {packages:['wordtree']});
        google.charts.setOnLoadCallback(drawChart);

    }

    _createUI();
}
