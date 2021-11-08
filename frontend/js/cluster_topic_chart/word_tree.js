function WordTree(cluster_groups, cluster_data, doc_data) {
    const width = 600;
    const height = 600;
    const total_num_docs = doc_data.length;
    let data_table;// Store the relation between clusters and topics
    let available_topics = []; // Store all the top 10 cluster topics
    let word_tree;  // word tree chart
    // Convert cluster topic to data table for Google chart
    function convert_cluster_topics_data_table() {
        data_table = new google.visualization.DataTable();
        data_table.addColumn('number', 'id');
        data_table.addColumn('string', 'childLabel');
        data_table.addColumn('number', 'parent');
        data_table.addColumn('number', 'weight');
        data_table.addColumn({role: 'style'});    // Word color
        // data_table.addColumn({type: 'string', role: 'tooltip'});

        let rows = [
            [0, 'Corpus', -1, total_num_docs, 'black']
        ];
        const root_id = 0;
        let id = 0;
        // Go through cluster groups
        for (const group of cluster_groups) {
            const g_id = ++id;
            const clusters = group['clusters'];
            const group_node = [g_id, group['group'], root_id, clusters.length, 'black'];
            for (const cluster_no of clusters) {
                const cluster = cluster_data.find(c => c['Cluster'] === cluster_no);
                const c_id = ++id;
                const cluster_topics = cluster['TopicN-gram'].slice(0, 10);    // Get top 10 topics
                cluster_topics.sort((a, b) => b['doc_ids'].length - a['doc_ids'].length);   // Sort cluster topics by number of doc
                const c_name = 'Cluster #' + cluster['Cluster'];
                const c_total = cluster['DocIds'].length;
                const cluster_node = [c_id, c_name, g_id, c_total, 'blue'];
                rows.push(cluster_node);
                // Get the sum of topic document counts
                const sub_total = cluster_topics.reduce((pre, cur) => pre + cur['doc_ids'].length, 0);
                for (const c_topic of cluster_topics) {
                    const t_id = ++id;
                    let proportion = c_topic['doc_ids'].length / sub_total * c_total;
                    const topic_node = [t_id, c_topic['topic'], c_id, proportion, 'green'];
                    rows.push(topic_node);
                }
                // Concatenate the cluster topics to available topics
                available_topics = available_topics.concat(cluster_topics);
            }
            rows.push(group_node);
        }
        // Add all rows
        data_table.addRows(rows);
        console.log(available_topics);
    }

    // Word tree select event
    function selectHandler() {
        const selectedItem = word_tree.getSelection();
        const word = selectedItem['word'];
        const color = selectedItem['color'];
        if (word.includes('#')) {
            // Cluster node
            const cluster_no = parseInt(word.split("#")[1]);
            // console.log("Cluster no = " + cluster_no );
        } else {
            // Topic
            const selected_topic = available_topics.find(t => t['topic'] === word);
            if (selected_topic) {
                // console.log("Topic = " + topic);
                // console.log(doc_data);
                const doc_ids = selected_topic['doc_ids'];
                const topic_docs = doc_data.filter(d => doc_ids.includes(d['DocId']));
                console.log(topic_docs);
                const doc_list = new DocList(topic_docs, selected_topic);
            }
        }

    }

    // Draw the chart
    function drawChart() {
        convert_cluster_topics_data_table();
        word_tree = new google.visualization.WordTree(document.getElementById('cluster_topic_chart'));
        // Draw the chart
        word_tree.draw(data_table, {
            maxFontSize: 18,
            wordtree: {
                width: width,
                height: height,
                format: 'explicit',
                type: 'suffix',
                forceIFrame: true,
            },
            tooltip: {trigger: 'selection'}
        });

        // Add the select handler
        google.visualization.events.addListener(word_tree, 'select', selectHandler);

    }

    // Add the instruction dialog
    function _createDialog(){
        $('#instruction').empty();
        $('#instruction').append($('<p>Clicking on a topic (green words) displays the associated articles within the cluster.</p>'));
        $('#instruction').dialog({
            modal: true,
            buttons: {
                Ok: function () {
                    $(this).dialog("close");
                }
            }
        });
        $("#opener").on("click", function () {
            $("#instruction").dialog("open");
        });
    }

    function _createUI() {
        _createDialog();
        // Set width and height
        $('#cluster_topic_chart').width(width).height(height);

        // Create the word tree
        google.charts.load('current', {packages: ['wordtree']});
        google.charts.setOnLoadCallback(drawChart);

    }

    _createUI();
}
