function ChartDocView(is_hide, cluster_approach, cluster_chart_data, cluster_topic_words) {
    // Create a scatter chart and cluster-topic word view
    function _createUI() {
         // Create the nodes and links of term occurrences.
        let d3_scatter_graph = new ScatterGraph(is_hide, cluster_approach, cluster_chart_data, cluster_topic_words);
        // Create a summary list view for
        let cluster_list_view = new ClusterListView(cluster_approach, cluster_topic_words);
    }

    _createUI();
}
